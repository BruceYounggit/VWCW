import datetime
import os
import time

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
from multiview_detector.utils.person_help import *


# device_n='cuda:0'
# 2023-1-11 amend
# class tranform_MLP(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(tranform_MLP, self).__init__()
#         self.mlp=nn.Sequential()
class DPerspTransDetector(nn.Module):
    def __init__(self, dataset, arch='resnet18', **kwargs):
        super().__init__()
        self.device = ['cuda:0', 'cuda:0']
        self.num_cam = dataset.num_cam
        self.input_feature = [1, 380, 676, 1]
        self.hfwf = dataset.hfwf
        self.hgwg = dataset.hgwg
        self.img_shape = dataset.img_shape
        # self.person_heights = range(1600, 2000, 100)  # [1600, 1700, 1800, 1900]
        self.person_heights = kwargs['person_heights']
        input_size = [1, self.hfwf[0], self.hfwf[1], 1]
        self.STN = SpatialTransformer_v3(input_size=input_size, output_size=self.hgwg,
                                         device=self.device[1],
                                         person_heights=kwargs['person_heights'])  # single-height

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

        self.img_classifier = nn.Sequential(
            nn.Conv2d(out_channel, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 1, bias=False)).to(self.device[1])

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

        self.confidence_attention = nn.Sequential(
            nn.Conv2d(1, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 16, 1), nn.ReLU(),
            nn.Conv2d(16, 1, 1, bias=False), nn.ReLU(),
        ).to(self.device[1])
        #
        # self.fusion_net = nn.Sequential(
        #     nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
        #     nn.Conv2d(1, 256, 3, padding=1), nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        #     nn.Conv2d(256, 1, 1, bias=False)
        # ).to(self.device[1])
        # max_pooling
        self.fusion_net = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1))

        self.transform_MLP = nn.Sequential(nn.Conv2d(1, out_channel, 3, padding=1), nn.ReLU(),
                                           nn.Conv2d(out_channel, out_channel, 3, padding=1)).to(self.device[1])

        # self.transform_MLP = nn.Sequential(nn.Linear(1, 128), nn.ReLU(),
        #                                    nn.Linear(128, 256), nn.ReLU(),
        #                                    nn.Linear(256, out_channel)).to(self.device[1])

        self.weight_calculation = nn.Sequential(
            nn.Conv2d(out_channel * 2, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 1, 3, padding=1), nn.ReLU(),
        ).to(self.device[1])

        if kwargs['fix_weight'] == 1:
            for param in self.confidence_attention.parameters():
                param.requires_grad = False

        self.cam_sel = camera_sel_fusion_layer_rbm2_full_szie(view_size=self.num_cam)
        Initialize_net(self.weight_calculation, mode='kmh')
        Initialize_net(self.transform_MLP, mode='kmh')
        Initialize_net(self.weight_calculation, mode='kmh')
        Initialize_net(self.confidence_attention,mode='kmh')

        # self.view_pooling_layer = view_pooling_layer_full_size(batch_size=1, view_size=self.num_cam)
        if kwargs['fix_2D'] == 1:
            if self.arch == 'resnet18':
                for param in self.base_pt1.parameters():
                    param.requires_grad = False
                for param in self.base_pt2.parameters():
                    param.requires_grad = False
            else:
                for param in self.base_pt.parameters():
                    param.requires_grad = False
            for param in self.img_classifier.parameters():
                param.requires_grad = False
        if kwargs['fix_svp'] == 1:
            for param_3 in self.view_gp_decoder.parameters():
                param_3.requires_grad = False

    def forward(self, imgs, visualize=False):
        B, N, C, H, W = imgs.shape
        assert N == 3
        imgs = imgs.view(-1, C, H, W).to(self.device[0])
        # step1: Extract feature of imgs
        if self.arch == 'resnet18':
            img_feature = self.base_pt1(imgs).to(self.device[1])
            img_feature = self.base_pt2(img_feature)
        elif self.arch == 'vgg16':
            img_feature = self.base_pt(imgs)
        else:
            raise Exception('Wrong arch.')
        img_feature = F.interpolate(img_feature, self.hfwf, mode='bilinear')
        # 2D img_decoding
        img_res = self.img_classifier(img_feature)
        world_feature = self.STN(img_feature.permute(0, 2, 3, 1), height=1750).permute(0, 3, 1, 2)

        # # weight_mask
        world_depth_maps = self.STN(self.depth_map.unsqueeze(3), height=1750).permute(0, 3, 1, 2)
        # b, c, h, w = world_depth_maps.size()
        # weight = self.cam_sel(world_depth_maps)
        # weight_mask_sum = torch.sum(world_depth_maps, dim=0, keepdim=True) + 1e-18
        # weight_mask_norm = torch.div(weight_mask_sum, weight_mask_sum)
        # world_feature_masked = torch.mul(world_feature, weight_mask_norm)
        # world_feature_fusion = torch.sum(world_feature_masked, dim=0, keepdim=True)
        weight_sum = torch.sum(world_depth_maps, dim=0, keepdim=True) + 1e-18
        weight_sum_morm = torch.div(world_depth_maps, weight_sum)
        weight_sum_morm = self.confidence_attention(weight_sum_morm)
        # w_mask = self.confidence_attention(w_mask)

        # world_depth_maps = self.transform_MLP(world_depth_maps.view(b * h * w, c)).view(b, -1, h, w) # for linear
        # world_depth_maps = self.transform_MLP(world_depth_maps)  # for conv
        # feat_depth_fusion = torch.cat([world_feature, world_depth_maps], dim=1)
        # weight = self.weight_calculation(feat_depth_fusion)
        # weight = torch.nn.functional.softmax(weight, dim=0)
        # feature * weight
        feature_proj123_masked = torch.multiply(world_feature, weight_sum_morm)
        feature_proj123_masked_pooled = torch.sum(feature_proj123_masked, dim=0, keepdim=True)
        feat_fusion_out = self.GP_Decoder(feature_proj123_masked_pooled).float()

        # single-view prediction * weight
        view_gp_output = self.view_gp_decoder(world_feature)  # single-view dmap, fixed
        view_gp_output_masked = torch.multiply(view_gp_output, weight_sum_morm)
        pred_fusion_out = torch.sum(view_gp_output_masked, dim=0, keepdim=True)

        # joint_fusion = torch.cat([feat_fusion_out, pred_fusion_out], dim=1)
        # joint_fusion = self.fusion_net(joint_fusion)
        # joint_fusion = torch.rand(*self.hgwg)
        joint_fusion = (feat_fusion_out + pred_fusion_out) / 2.0
        return img_res, view_gp_output, feat_fusion_out, pred_fusion_out, joint_fusion, weight_sum_morm


if __name__ == '__main__':
    print(datetime.datetime.now())
    from multiview_detector.datasets.CityStreet.framedataset_depthmap import frameDataset_depth_map_full_size
    from multiview_detector.datasets.CityStreet.Citystreet import Citystreet
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    img_reduce = 2
    world_reduce = 2
    train_trans = T.Compose([T.Resize([1520 // img_reduce, 2704 // img_reduce]), T.ToTensor(), normalize])
    t0 = time.time()
    dataset_train = frameDataset_depth_map_full_size(Citystreet(os.path.expanduser('~/Data/CityStreet')), train=True,
                                                     transform=train_trans,
                                                     world_reduce=world_reduce, map_sigma=3, facofmaxgt=1000,
                                                     facofmaxgt_gp=10)
    # dataset_test = frameDataset(Citystreet(os.path.expanduser('~/Data/CityStreet')), train=False, map_sigma=5)
    # imgs, detec_map_gt, hw_random, frame = dataset_train.__getitem__(0)
    dataloader = DataLoader(dataset_train, 1, False, num_workers=4)
    imgs, imgs_gt, masked_view_gp_gt, gp_gt, frame = next(iter(dataloader))
    t1 = time.time()
    print(f't1-t0={t1 - t0:.3f}')
    # imgs [1,3,3,760,1352], imgs_gt is list: 3x[1,1,380,676], detect_map_gt is list: 2x[1,200,200]
    # count_map_gt is list:2x[1,200,200], view_mask is dict: 3X[1,768,640], hw_random is tensor:[1,2,2]
    # print('imgs shape', imgs.shape)
    # person_heights = [1600, 1700, 1800, 1900]
    person_heights = [1750]
    model = DPerspTransDetector(dataset_train, fix_2D=1, fix_svp=1, fix_weight=1, person_heights=person_heights)
    t3 = time.time()
    model = loadmodel(model,
                      model_dir='/home/yunfei/Study/MVD_VCW/logs/citystreet_frame/resnet18/2D_SVP_WS/2023-04-24_17-00-58_fix2D1.0w1_fixsvp1.0w1_momentum0.9_weight_decay0.0001_lr0.01_lrsonecycle_epo100_ct0.4_nt10_dt20/MultiviewDetector_100.pth')

    img_res, view_gp_output, feat_fusion_out, pred_fusion_out, weight = model(imgs, visualize=True)
    t4 = time.time()
    print(f'Time for completing projection is {t4 - t3:.3f}.')
