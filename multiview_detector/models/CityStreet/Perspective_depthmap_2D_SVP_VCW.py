import datetime
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.vgg import vgg11, vgg16
from multiview_detector.models.resnet import resnet18
from multiview_detector.models.CityStreet.spatial_transformer import SpatialTransformer_v3
from multiview_detector.utils.person_help import *


# weight = torch.full((3, 1, 384, 320), 0.333)
class DPerspTransDetector(nn.Module):
    def __init__(self, dataset, weight, arch='resnet18', **kwargs):
        super().__init__()
        self.device = ['cuda:0', 'cuda:0']
        self.num_cam = dataset.num_cam
        self.weight = nn.Parameter(weight)

        self.input_feature = [1, 380, 676, 1]
        self.hfwf = dataset.hfwf
        self.hgwg = dataset.hgwg
        self.img_shape = dataset.img_shape
        self.person_heights = kwargs['person_heights']
        input_size = [1, self.hfwf[0], self.hfwf[1], 1]
        self.STN = SpatialTransformer_v3(input_size=input_size, output_size=self.hgwg,
                                         device=self.device[1],
                                         person_heights=kwargs['person_heights'])  # single-height
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

        # self.confidence_attention = nn.Sequential(nn.Conv2d(1, 256, 3, padding=1), nn.ReLU(),
        #                                           nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        #                                           nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
        #                                           nn.Conv2d(128, 1, 1), nn.ReLU()).to(self.device[1])

        self.weight_consistency = nn.Sequential(nn.Conv2d(1, 64, 1), nn.ReLU(),
                                                nn.Conv2d(64, 1, 1, bias=False)).to(self.device[1])
        Initialize_net(self.weight_consistency, mode='hkm')
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

    def forward(self, imgs, gp_gt):
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
        world_feature = self.STN(img_feature.permute(0, 2, 3, 1), height=self.person_heights[0]).permute(0, 3, 1, 2)
        view_gp_output = self.view_gp_decoder(world_feature)
        # if weight is not None:
        view_fusion = torch.mul(world_feature, self.weight.to(world_feature.device))
        view_fusion = torch.sum(view_fusion, dim=0, keepdim=True)
        map_res = self.GP_Decoder(view_fusion)

        # After the fowardbroadcast, update the weight
        real_weight = []
        mask_map_res = torch.mul(map_res, torch.norm(world_feature, dim=1, keepdim=True) > 0)
        mask_gp_gt = torch.mul(gp_gt.to(world_feature.device), torch.norm(world_feature, dim=1, keepdim=True) > 0)
        diff = mask_gp_gt - mask_map_res
        real_weight = self.weight_consistency(diff)

        # real_weight = torch.cat(real_weight, dim=0)
        real_weight = torch.mul(real_weight, torch.norm(world_feature, dim=1, keepdim=True) > 0)
        weight_sum = torch.sum(real_weight, dim=0, keepdim=True)
        real_weight = torch.div(real_weight, weight_sum + 1e-18)
        self.weight.data = real_weight
        self.weight = self.weight.to(self.device[1])
        return img_res, view_gp_output, map_res


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
    weight = torch.ones(3, 1, 384, 320) * 1 / 3
    model = DPerspTransDetector(dataset_train, arch='vgg16', fix_2D=1, fix_svp=1, fix_weight=1,
                                person_heights=person_heights, weight=weight)
    t3 = time.time()
    model = loadmodel(model, model_dir='/home/yunfei/Study/MVD_VCW/logs/citystreet_frame/'
                                       'vgg16/2D_SVP_VCW/2023-05-23_15-56-08_fix2D1.0w1_fixsvp0.0w1_'
                                       'fixconf0.0_momentum0.9_weight_decay0.0001_lr0.01_lrsonecycle_epo200'
                                       '_ct0.4_nt10_dt20/MultiviewDetector.pth')

    img_res, view_gp_out, map_res= model(imgs, gp_gt)
    t4 = time.time()
    print(f'Time for completing projection is {t4 - t3:.3f}.')
