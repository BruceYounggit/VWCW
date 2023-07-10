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
from multiview_detector.utils.person_help import vis, Initialize_net


# device_n='cuda:0'
# 2023-1-11 amend
class DPerspTransDetector(nn.Module):
    def __init__(self, dataset, arch='resnet18', **kwargs):
        super().__init__()
        self.device = ['cuda:0', 'cuda:0']
        self.num_cam = dataset.num_cam
        self.input_feature = [1, 380, 676, 1]
        self.hfwf = dataset.hfwf
        self.hgwg = dataset.hgwg
        self.img_shape = dataset.img_shape

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
            nn.Conv2d(64, 1, 1, bias=False)).to(self.device[0])



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

        return img_res

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

    dataloader = DataLoader(dataset_train, 1, False, num_workers=4)
    imgs, imgs_gt, masked_view_gp_gt, gp_gt, frame = next(iter(dataloader))
    t1 = time.time()
    print(f't1-t0={t1 - t0:.3f}')
    model = DPerspTransDetector(dataset_train)
    t3 = time.time()
    img_res = model(imgs, visualize=True)
