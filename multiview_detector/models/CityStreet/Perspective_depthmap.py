import datetime
import os
import time

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '3,1'
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision.models.vgg import vgg11, vgg16
from multiview_detector.models.resnet import resnet18
from multiview_detector.models.CityStreet.spatial_transformer import SpatialTransformer_v3
import matplotlib.pyplot as plt
from multiview_detector.models.CityStreet.processing_layer import camera_sel_fusion_layer_rbm2_full_szie, \
    view_pooling_layer, Dmap_consist_loss_layer
import torch.nn.init as init


# device_n='cuda:0'
# 2023-1-11 amend
class DPerspTransDetector(nn.Module):
    def __init__(self, dataset, arch='vgg16', **kwargs):
        super().__init__()
        self.device = ['cuda:0', 'cuda:0']
        self.num_cam = dataset.num_cam
        self.input_feature = [1, 380, 676, 1]
        self.hfwf = dataset.hfwf
        self.hgwg = dataset.hgwg
        self.img_shape = dataset.img_shape
        # self.person_heights = range(1600, 2000, 100)  # [1600, 1700, 1800, 1900]
        self.person_heights = kwargs['person_heights']
        self.fix_weightCNN = kwargs['fix_weightCNN']
        self.depth_scales = len(self.person_heights)
        self.view_masks = dataset.view_masks
        input_size = [1, self.hfwf[0], self.hfwf[1], 1]
        self.SpatialTransInstance = SpatialTransformer_v3(input_size=input_size, output_size=self.hgwg,
                                                          device=self.device[0],
                                                          person_heights=[1750])  # single-height
        self.w_mask = None
        # Normalize depth map before confidence layer as input.
        self.depth_map = dataset.depth_map
        self.arch = arch
        if arch == 'vgg11':
            base = vgg11().features
            base[-1] = nn.Sequential()
            base[-4] = nn.Sequential()
            split = 10
            self.base_pt1 = base[:split].to(self.device[0])
            self.base_pt2 = base[split:].to(self.device[1])
            out_channel = 512
        elif arch == 'resnet18':
            base = nn.Sequential(*list(resnet18(
                replace_stride_with_dilation=[False, True, True]).children())[:-2])
            split = 7
            self.base_pt1 = base[:split].to(self.device[0])
            self.base_pt2 = base[split:].to(self.device[1])
            out_channel = 512
        elif arch == 'vgg16':
            base = vgg16().features
            split = 16  # 7 conv layers, and 2 maxpooling layers.
            self.base_pt = base[:split].to(self.device[0])
            out_channel = 256
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')
        # 2.5cm -> 0.5m: 20x
        # self.img_classifier = nn.Sequential(nn.Conv2d(out_channel, 64, 1), nn.ReLU(),
        #                                     nn.Conv2d(64, 1, 1, bias=False)).to(self.device[0])
        self.img_decoder = nn.Sequential(
            nn.Conv2d(out_channel, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 1, bias=False)).to(self.device[0])

        # out_channel = 256
        # SVP
        self.view_gp_decoder = nn.Sequential(nn.Conv2d(out_channel, 512, 3, padding=1), nn.ReLU(),
                                             nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                                             nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                                             nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
                                             nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                                             nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                                             nn.Conv2d(64, 1, 1, bias=False)).to(self.device[1])
        # GP decoder
        self.GP_Decoder = nn.Sequential(nn.Conv2d(out_channel, 512, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(64, 1, 1, bias=False)).to(self.device[1])

        # self.coord_map = nn.Parameter(self.coord_map, False).to(self.device[1])
        self.confidence_attention = nn.Sequential(
            nn.Conv2d(1, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 16, 1), nn.ReLU(),
            nn.Conv2d(16, 1, 1, bias=False), nn.ReLU(),
        ).to(self.device[1])
        self.cam_sel = camera_sel_fusion_layer_rbm2_full_szie(view_size=self.num_cam)
        self.Dmap_consist_loss_layer = Dmap_consist_loss_layer()
        # Initialization
        self.Initialize_net()

        # self.view_pooling_layer = view_pooling_layer_full_size(batch_size=1, view_size=self.num_cam)
        if kwargs['fix_2D'] == 1:
            for param_1 in self.base_pt.parameters():
                param_1.requires_grad = False
            for param_2 in self.img_classifier.parameters():
                param_2.requires_grad = False
        if kwargs['fix_svp'] == 1:
            for param_3 in self.map_classifier.parameters():
                param_3.requires_grad = False

        # First, fix self.confidence CNN
        if self.fix_weightCNN == 1:
            for param_4 in self.confidence_attention.parameters():
                param_4.requires_grad = False

    # Single_height projection
    # def projection_layer(self, x, hw_random, default_height=1750):

    def Initialize_net(self):
        for layer in self.confidence_attention.modules():
            if isinstance(layer, nn.Conv2d):
                # init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                fan_out = layer.out_channels
                init.constant_(layer.weight, 1 / fan_out)
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)

        for layer in self.GP_Decoder.modules():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                # init.constant_(layer.weight, 1)
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)

    # @staticmethod
    # def visualize(self, x, title='None'):
    #     pixel_max = x.max()
    #     pixel_min = x.min()
    #     norm = matplotlib.colors.Normalize(vmin=pixel_min, vmax=pixel_max)
    #     fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
    #     fig.suptitle(title)
    #     for i in range(self.num_cam):
    #         mappable_i = matplotlib.cm.ScalarMappable(norm=norm)
    #         im = axs[i].imshow(x[i].detach().cpu().squeeze(), norm=norm)
    #         # axs[i].set_xticks([])
    #         # axs[i].set_yticks([])
    #         axs[i].set_title(f'View_{i + 1}')
    #     fig.colorbar(mappable_i, ax=axs)
    #     plt.show()
    #     plt.close(fig)

    def forward(self, imgs, visualize=False):
        B, N, C, H, W = imgs.shape
        assert N == 3
        imgs = imgs.view(-1, C, H, W).to(self.device[0])
        # step1: Extract feature of imgs
        if self.arch == 'resnet18':
            img_feature = self.base_pt1(imgs)
            img_feature = self.base_pt2(img_feature)
        elif self.arch == 'vgg16':
            img_feature = self.base_pt(imgs)
        else:
            raise Exception('Wrong arch.')
        img_feature = F.interpolate(img_feature, self.hfwf, mode='bilinear', align_corners='True')
        # 2D img_decoding
        img_res = self.img_decoder(img_feature)

        world_features = self.STN(img_feature.permute(0, 2, 3, 1), height=1750).permute(0, 3, 1, 2)
        # if self.fix_weightCNN:
        #     if self.w_mask is None:
        #         world_depth_maps = self.STN(self.depth_map.unsqueeze(3), height=1750).permute(0, 3, 1, 2)
        #         if visualize:
        #             self.visualize(world_depth_maps, title='Projection depth map after Normalization')
        #         # step4: Weight mask [6,384,320,256]
        #         w_mask_cnn = self.confidence_attention(world_depth_maps)
        #         self.w_mask = self.cam_sel(w_mask_cnn)
        #     w_mask = self.w_mask
        # else:
        world_depth_maps = self.STN(self.depth_map.unsqueeze(3), height=1750).permute(0, 3, 1, 2)
        w_mask_cnn = self.confidence_attention(world_depth_maps)
        w_mask = self.cam_sel(w_mask_cnn)

        feature_proj123_masked = torch.multiply(world_features, w_mask)
        feature_proj123_masked_pooled = torch.sum(feature_proj123_masked, dim=0, keepdim=True)
        x_output = self.GP_Decoder(feature_proj123_masked_pooled).float()

        view_gp_output = self.view_gp_decoder(world_features)  # single-view dmap, fixed
        view_gp_output_masked = torch.multiply(view_gp_output, w_mask)
        view_gp_output_masked_pooled = torch.sum(view_gp_output_masked, dim=0, keepdim=True)
        if visualize:
            self.visualize(view_gp_output, title='view_gp_output')
            self.visualize(view_gp_output_masked, title='view_gp_output_masked')
            plt.imshow(x_output.detach().cpu().squeeze())
            plt.colorbar()
            plt.show()
            plt.imshow(view_gp_output_masked_pooled.detach().cpu().squeeze())
            plt.colorbar()
            plt.show()
        # y_output = self.GP_Decoder(view_gp_output_masked_pooled)
        Dmap_consist_loss = self.Dmap_consist_loss_layer([x_output, view_gp_output_masked_pooled])
        mean_out = (x_output + view_gp_output_masked_pooled) / 2

        return img_res, view_gp_output, mean_out, w_mask, Dmap_consist_loss


if __name__ == '__main__':
    print(datetime.datetime.now())
    from multiview_detector.datasets.CityStreet.framedataset_depthmap import frameDataset_depth_map_full_size
    from multiview_detector.datasets.CityStreet.Citystreet import Citystreet
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    img_reduce = 2

    world_reduce = 2
    transform = T.Compose([T.Resize([1520 // img_reduce, 2704 // img_reduce]),
                           T.ToTensor(),
                           T.Normalize((0.442, 0.429, 0.409), (0.250, 0.260, 0.255))])
    t0 = time.time()
    dataset = frameDataset_depth_map_full_size(Citystreet(os.path.expanduser('~/Data/CityStreet')), transform=transform,
                                               facofmaxgt=1000, facofmaxgt_gp=10, world_reduce=world_reduce,
                                               img_reduce=img_reduce)
    t1 = time.time()
    print('Time for preparing data:', t1 - t0)
    dataloader = DataLoader(dataset, 1, False, num_workers=4)
    imgs, detec_map_gt, frame = next(iter(dataloader))
    # imgs [1,3,3,760,1352], imgs_gt is list: 3x[1,1,380,676], detect_map_gt is list: 2x[1,200,200]
    # count_map_gt is list:2x[1,200,200], view_mask is dict: 3X[1,768,640], hw_random is tensor:[1,2,2]
    # print('imgs shape', imgs.shape)
    # person_heights = [1600, 1700, 1800, 1900]
    person_heights = [0, 1750]
    model = DPerspTransDetector(dataset, fix_2D=1, fix_svp=1, fix_weightCNN=0, person_heights=person_heights)
    t3 = time.time()

    x_output, view_gp_out, w_mask = model(imgs, visualize=True, pretrained_flag=False)
    print(model.intermediate_output)
    t4 = time.time()
    plt.imshow(x_output.detach().cpu().squeeze())
    plt.colorbar()
    plt.show()

    print(f'Time for completing projection is {t4 - t3:.2f}.')
    # imgs_result is list: 3x[1,1,380,676], patch_view_dmaps is dict:2x[3,1,200,200],
    # map_patch_results is dict:2x[1,1,200,200]
    # print(imgs_result[0].shape, '\n', patch_view_dmaps[1])
    # loss, map_res = model(imgs, imgs_gt, gp_gt, 1, visualize=False)
    # # imgs_gt_trans = model.trans_img(imgs_gt[0][None])
    # # print('imgs_gt trans',imgs_gt_trans.shape)
    # print('loss :', loss)
    # print('map_res shape', map_res.shape)
    # plt.imshow(map_res.detach().squeeze().cpu().numpy())
    # plt.show()
    # print(datetime.datetime.now())
