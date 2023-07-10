import datetime
import os
import time

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
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

        # Initialize_net(self.view_gp_decoder)

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
        view_gp_output = self.view_gp_decoder(world_feature)  # single-view dmap, fixed

        return img_res, view_gp_output


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
    model = DPerspTransDetector(dataset_train, fix_2D=0, fix_svp=0, person_heights=person_heights)
    t3 = time.time()
    model = loadmodel(model, model_dir='/home/yunfei/Study/MVD_VCW/logs/citystreet_frame/vgg16/2D_SVP/2023-04-24_19-12-45_fix2D1.0w1_fixsvp0.0w1_moment'
                                       'um0.9_weight_decay0.0001_lr0.01_lrsonecycle_epo100_ct0.4_nt'
                                       '10_dt20/MultiviewDetector_100.pth')
    img_res, view_gp_output = model(imgs, visualize=True)
    t4 = time.time()
    print(f'Time for completing projection is {t4 - t3:.3f}.')
