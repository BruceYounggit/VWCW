import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torch.nn import init
from torchvision.models.vgg import vgg11, vgg16
from multiview_detector.models.resnet import resnet18
from multiview_detector.utils.person_help import *
import matplotlib.pyplot as plt


# os.environ['CUDA_VISIBLE_DEVICES']='3'

class PerspTransDetector_2D_SVP_3D(nn.Module):
    def __init__(self, dataset, arch='vgg16', **kwargs):
        super().__init__()
        self.device = kwargs['device']
        self.num_cam = dataset.num_cam
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.base.intrinsic_matrices,
                                                                           dataset.base.extrinsic_matrices,
                                                                           dataset.base.worldgrid2worldcoord_mat)
        # self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])
        # img
        self.upsample_shape = dataset.upsample_shape
        self.input_img_shape = dataset.input_img_shape
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        # # map
        map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))
        # # projection matrices: img feat -> map feat
        self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                          for cam in range(self.num_cam)]
        if arch == 'vgg16':
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

        self.view_gp_decoder = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
                                             nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                                             nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                                             nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
                                             nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                                             nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                                             nn.Conv2d(64, 1, 1, bias=False)).to(self.device[1])

        self.GP_decoder = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                                        nn.Conv2d(64, 1, 1, bias=False)).to(self.device[1])

        self.view_weight = nn.Sequential(nn.Conv2d(1, 256, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                                         nn.Conv2d(128, 1, 1, bias=False)).to(self.device[1])

        self.Initial_net(self.view_weight)
        # self.Initial_net(self.GP_decoder)
        if kwargs['fix_2D'] == 1:
            for param_1 in self.base_pt.parameters():
                param_1.requires_grad = False
            for param_2 in self.img_classifier.parameters():
                param_2.requires_grad = False
        if kwargs['fix_svp'] == 1:
            for param_3 in self.view_gp_decoder.parameters():
                param_3.requires_grad = False

    def Initial_net(self, *args, mode='equal'):
        import torch.nn.init as init
        if mode == 'equal':
            for net in args:
                for layer in net.modules():
                    if isinstance(layer, nn.Conv2d):
                        fan_out = layer.out_channels
                        init.constant(layer.weight, 1 / (fan_out * layer.kernel_size[0] * layer.kernel_size[1]))
                        # init.kaiming_normal_(layer.weight, nonlinearity='relu')
                        if layer.bias is not None:
                            init.constant(layer.bias, 0)
        elif mode == 'norm':
            for net in args:
                for layer in net.modules():
                    if isinstance(layer, nn.Conv2d):
                        init.kaiming_normal_(layer.weight, nonlinearity='relu')
                        if layer.bias is not None:
                            init.constant(layer.bias, 0)

    def forward(self, imgs, visualize=False):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        world_features = []
        world_gts = []
        imgs_result = []
        view_gp_results = []
        view_confidence_maps = []
        view_masks = []
        for cam in range(self.num_cam):
            img_feature = self.base_pt(imgs[:, cam].to(self.device[0])).to(self.device[1])
            # img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear').to(self.device[1])
            img_res = self.img_classifier(img_feature)
            imgs_result.append(img_res)
            proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().to(self.device[1])
            world_feature = kornia.geometry.warp_perspective(img_feature, proj_mat, self.reducedgrid_shape)
            all_one = torch.ones(1, 1, img_feature.shape[2], img_feature.shape[3]).to(self.device[0])
            view_mask = kornia.geometry.warp_perspective(all_one, proj_mat, self.reducedgrid_shape)
            view_gp_res = self.view_gp_decoder(world_feature.to(self.device[1]))
            view_masks.append(view_mask)
            view_gp_results.append(view_gp_res)
            world_features.append(world_feature)
            if visualize and cam == 0:
                vis(torch.norm(img_feature.detach(), dim=1).cpu().squeeze())
                vis(img_res.squeeze().cpu())
                vis(torch.norm(world_feature.detach(), dim=1).cpu().squeeze())
        imgs_result = torch.cat(imgs_result)
        world_features = torch.cat(world_features)
        view_masks = torch.cat(view_masks)
        view_gp_results = torch.cat(view_gp_results, dim=0)
        weight = self.view_weight(view_gp_results)
        # weight_exp = torch.exp(weight)
        weight_mask = torch.mul(weight, view_masks)
        weight_mask_sum = torch.sum(weight_mask, dim=0, keepdim=True) + 1e-18
        weight_mask_norm = torch.div(weight_mask, weight_mask_sum)
        world_features_with_weight = torch.mul(world_features, weight_mask_norm)
        world_features_with_weight_dimsum = torch.sum(world_features_with_weight, dim=0, keepdim=True)
        map_res = self.GP_decoder(world_features_with_weight_dimsum)
        if visualize:
            vis(view_gp_results[0])
            vis(weight_mask[0])
            vis(weight_mask[0])
            vis(torch.norm(world_features_with_weight[0], dim=0))
            # vis(map_res)
        return imgs_result, view_gp_results, map_res, view_masks, weight

    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)

            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
            # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
            permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
            pass
        return projection_matrices


def test():
    from multiview_detector.datasets.frameDataset import frameDataset
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.datasets.MultiviewX import MultiviewX
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    transform = T.Compose([T.Resize([720, 1280]),  # H,W
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), transform=transform)
    dataloader = DataLoader(dataset, 1, False, num_workers=0)
    imgs, map_gt, imgs_gt, frame = next(iter(dataloader))

    model = PerspTransDetector_2D_SVP_3D(dataset, fix_2D=1, fix_svp=1)
    pre_model_dir = '/home/yunfei/Study/Baseline_dataset_multiviewX_WildTrack/logs/multiviewx_frame/2D_SVP' \
                    '/2023-03-09_11-57-12_fix2D0w1_fixsvp0w1_momentum0.9_weight_decay0.0005_lr0.01_lrsonecycle_epo200_ct0.4_' \
                    'nt40_dt80/MultiviewDetector.pth'
    model = loadmodel(model, pre_model_dir)

    imgs_result, view_gp_results, map_res, weight_exp = model(imgs, imgs_gt, visualize=True)
    pass


if __name__ == '__main__':
    test()
