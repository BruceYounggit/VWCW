import os
import random

import h5py
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor

from multiview_detector.datasets.CityStreet.datagen import conv_process
from multiview_detector.datasets.CityStreet.view_mask import under_cropped_mask, get_view_gp_mask
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.utils.gaussian_blur_detecting import draw_umich_gaussian, gaussian2D
# from multiview_detector.utils.gaussian_blur_counting import draw_umich_gaussian, gaussian2D


class frameDataset_depth_map_full_size(VisionDataset):
    def __init__(self, base, train=True, transform=ToTensor(), target_transform=ToTensor(), map_sigma=3, world_reduce=2,
                 img_reduce=2, **kwargs):
        # Totensor() Convert a PIL Image or numpy.ndarray to tensor. This transform does not support torchscript.
        super().__init__(base.root, transform=transform, target_transform=target_transform)
        # self.mapgtfromcoord = kwargs['mapgtfromcoord']
        self.map_sigma = map_sigma
        self.gaussian_kernel_sum = gaussian2D(shape=[6 * map_sigma + 1, 6 * map_sigma + 1], sigma=map_sigma).sum()
        self.img_shape = base.img_shape
        self.train = train
        self.base = base
        self.root, self.num_frame = base.root, base.num_frame
        self.ground_plane_shape = base.worldgrid_shape
        self.hfwf = (380, 676)
        self.img_reduce = img_reduce
        self.world_reduce = world_reduce
        self.hgwg = tuple(map(lambda x: int(x / self.world_reduce), self.ground_plane_shape))
        self.num_cam = base.num_cam
        self.facofmaxgt = kwargs['facofmaxgt']
        self.facofmaxgt_gp = kwargs['facofmaxgt_gp']

        view_masks = []
        for view in range(1, self.num_cam + 1):
            view_masks.append(get_view_gp_mask(view, self.hgwg))
        self.view_masks = np.stack(view_masks)

        self.depth_map = self.get_depth_maps()

        self.transform = transform
        frame_rangelist = [range(636, 1236, 2), range(1236, 1636, 2)]
        if self.train:
            self.frame_range = frame_rangelist[0]
        else:
            self.frame_range = frame_rangelist[1]
        self.img_fpaths = self.base.get_img_fpath()
        self.map_gt_from_coords = {}
        self.masked_view_gp_gt = {_: [] for _ in self.frame_range}
        self.map_gt_from_density_maps = {}
        self.imgs_head_gt = {view: {} for view in range(1, 4)}

        self.download(self.frame_range)  # 获得map_gt,和 imgs_gt

        self.gt_fpath = os.path.join(self.root, 'gt_pixel_0.1m.txt')
        if not os.path.exists(self.gt_fpath):
            self.prepare_gt()

    # 得到每个image中所有人的map_gt[frame],imgs_head_foot[frame]，city数据集下只考虑它的头所在位置，
    def get_dmaps_path(self):
        aimdir = {
            'gp_train': os.path.join(self.root, 'GT_density_maps/'
                                                'ground_plane/train/Street_groundplane_train_dmaps_10.h5'),
            'gp_test': os.path.join(self.root, 'GT_density_maps/'
                                               'ground_plane/test/Street_groundplane_test_dmaps_10.h5'),
            'v1_train': os.path.join(self.root, 'GT_density_maps/camera_view/train/Street_view1_dmap_10.h5'),
            'v2_train': os.path.join(self.root, 'GT_density_maps/camera_view/train/Street_view2_dmap_10.h5'),
            'v3_train': os.path.join(self.root, 'GT_density_maps/camera_view/train/Street_view3_dmap_10.h5'),

            'v1_test': os.path.join(self.root, 'GT_density_maps/camera_view/test/Street_view1_dmap_10.h5'),
            'v2_test': os.path.join(self.root, 'GT_density_maps/camera_view/test/Street_view2_dmap_10.h5'),
            'v3_test': os.path.join(self.root, 'GT_density_maps/camera_view/test/Street_view3_dmap_10.h5')}
        return aimdir

    def load_h5(self, h5dir, frame_range_list):
        temp_gt = {}
        with h5py.File(h5dir, 'r') as fp:
            dmap_i = fp['density_maps']
            dmap_i = np.squeeze(dmap_i).astype(np.float32)
            # print('dmap_i shape', dmap_i.shape)
            for i in range(0, dmap_i.shape[0]):
                temp_gt[frame_range_list[i]] = dmap_i[i][:][:]
        return temp_gt

    # 因为每帧图像对应的裁剪位置不一样，则有不同的gt坐标，所以这个总的gt文件要在trainer.py中完成
    # gp_map 就是对(os.path.expanduser('~/Data/CityStreet/Street_groundplane_pmap.h5')提取。
    def prepare_gt(self):
        og_gt = []
        with h5py.File(os.path.expanduser('~/Data/CityStreet/Street_groundplane_pmap.h5'), 'r') as f:
            for i in range(f['v_pmap_GP'].shape[0]):
                singlePerson_Underframe = f['v_pmap_GP'][i]
                frame = int(singlePerson_Underframe[0])
                # personID = int(singlePerson_Underframe[1])
                # 原论文grid_x 为H这条边，singleFrame_underframe[3]指的是cy，最大值不超过768
                # 原论文grid_y 为W这条边，singleFrame_underframe[2]指的是cx，最大值不超过640
                grid_y = int(singlePerson_Underframe[2])  # 乘以4之后，每个pixel代表2.5cm [现在不乘]
                grid_x = int(singlePerson_Underframe[3])  # [3072, 2560]
                # height = int(singlePerson_Underframe[4])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    # og_gt = np.stack(og_gt, axis=0)
    # os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
    # np.savetxt(self.gt_fpath, og_gt, '%d')
    def get_depth_maps(self):
        depth_maps_dir = '/home/yunfei/Data/CityStreet/ROI_maps/Distance_maps'
        depth_map_dir_list = sorted(os.listdir(depth_maps_dir))
        depth_map_array_list = []
        for i_dir in depth_map_dir_list:
            a = np.load(os.path.join(depth_maps_dir, i_dir))['arr_0']  # [380,676]
            depth_map_array_list.append(a)
        depth_map = np.stack(depth_map_array_list, axis=0)
        depth_map = torch.from_numpy(depth_map)
        # depth_map = torch.pow(depth_map, 0.25)
        # depth_map /= 1000
        # depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) + 1e-18
        # depth_map = -torch.log(depth_map)
        # depth_map = 1 - depth_map
        depth_map = depth_map.min() / depth_map
        return depth_map

    def download(self, frame_range):
        aimdir = self.get_dmaps_path()
        # map_gt [768,640]
        # if self.mapgtfromcoord is True:
        with h5py.File(os.path.expanduser('~/Data/CityStreet/Street_groundplane_pmap.h5'), 'r') as f:
            grounddata = f['v_pmap_GP']
            zero_array = np.zeros(self.hgwg)
            for frame in frame_range:
                occupancy_info = (grounddata[grounddata[:, 0] == frame, 2:4])
                occupancy_map = np.zeros(self.hgwg)
                view_masked_occupancy_maps = np.zeros((self.num_cam, *self.hgwg))
                for idx in range(occupancy_info.shape[0]):
                    cx, cy = occupancy_info[idx]
                    cx = int(cx // self.world_reduce)
                    cy = int(cy // self.world_reduce)
                    center = (cx, cy)
                    # create ground truth
                    occupancy_map = draw_umich_gaussian(occupancy_map, center, sigma=self.map_sigma)
                    # judge whether the point 'center' is within certain view-area on ground plane.
                    zero_array[cy, cx] = 10
                    for view in range(self.num_cam):
                        if (zero_array * self.view_masks[view]).max() == 10:
                            view_masked_occupancy_maps[view] = draw_umich_gaussian(view_masked_occupancy_maps[view],
                                                                                   center, self.map_sigma)
                    zero_array[cy, cx] = 0

                self.map_gt_from_coords[frame] = occupancy_map
                # view_gp_gt
                self.masked_view_gp_gt[frame] = torch.from_numpy(view_masked_occupancy_maps)

        # imgs_gt [380,676]
        if self.train:
            # temp_gp_train = self.load_h5(aimdir['gp_train'], frame_range)
            temp_view1_train = self.load_h5(aimdir['v1_train'], frame_range)
            temp_view2_train = self.load_h5(aimdir['v2_train'], frame_range)
            temp_view3_train = self.load_h5(aimdir['v3_train'], frame_range)
            for i in frame_range:
                # temp_gp_train[i] = conv_process(temp_gp_train[i][:, :, None], stride=self.world_reduce,
                #                                 filter_size=self.world_reduce)
                # self.map_gt_from_density_maps[i] = temp_gp_train[i]
                self.imgs_head_gt[1][i] = temp_view1_train[i]
                self.imgs_head_gt[2][i] = temp_view2_train[i]
                self.imgs_head_gt[3][i] = temp_view3_train[i]
        else:
            # temp_gp_test = self.load_h5(aimdir['gp_test'], frame_range)
            temp_view1_test = self.load_h5(aimdir['v1_test'], frame_range)
            temp_view2_test = self.load_h5(aimdir['v2_test'], frame_range)
            temp_view3_test = self.load_h5(aimdir['v3_test'], frame_range)
            for i in frame_range:
                # temp_gp_test[i] = conv_process(temp_gp_test[i][:, :, None], stride=self.world_reduce,
                #                                filter_size=self.world_reduce)
                # self.map_gt_from_density_maps[i] = temp_gp_test[i]
                self.imgs_head_gt[1][i] = temp_view1_test[i]
                self.imgs_head_gt[2][i] = temp_view2_test[i]
                self.imgs_head_gt[3][i] = temp_view3_test[i]

    def __getitem__(self, index):
        frame = self.frame_range[index]
        imgs = []
        for view in range(1, 4):
            fpath = self.img_fpaths[view][frame]
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)

        # img_gt
        imgs_gt = []
        for view in range(1, 4):
            img_gt = self.imgs_head_gt[view][frame] * self.facofmaxgt
            imgs_gt.append(img_gt[None])
        imgs_gt = torch.from_numpy(np.concatenate(imgs_gt))

        # ground plane gt
        gp_gt = torch.from_numpy(self.map_gt_from_coords[frame])[None] * self.facofmaxgt_gp

        # masked view_gp_gt on the ground plane
        masked_view_gp_gt = self.masked_view_gp_gt[frame] * self.facofmaxgt_gp

        return imgs, imgs_gt.float(), masked_view_gp_gt.float(), gp_gt.float(), frame

    def __len__(self):
        return len(self.frame_range)


if __name__ == '__main__':
    from multiview_detector.datasets.CityStreet.Citystreet import Citystreet
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from multiview_detector.utils.person_help import purevis, vis

    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    img_reduce = 2
    world_reduce = 2
    train_trans = T.Compose([T.Resize([1520 // img_reduce, 2704 // img_reduce]), T.ToTensor(), normalize])
    dataset_train = frameDataset_depth_map_full_size(Citystreet(os.path.expanduser('~/Data/CityStreet')), train=True,
                                                     transform=train_trans,
                                                     world_reduce=world_reduce, map_sigma=3, facofmaxgt=1000,
                                                     facofmaxgt_gp=10)
    # dataset_test = frameDataset(Citystreet(os.path.expanduser('~/Data/CityStreet')), train=False, map_sigma=5)
    # imgs, detec_map_gt, hw_random, frame = dataset_train.__getitem__(0)
    dataloader = DataLoader(dataset_train, 1, False, num_workers=4)
    imgs, imgs_gt, masked_view_gp_gt, gp_gt, frame = next(iter(dataloader))
    pass
    # with h5py.File(os.path.expanduser('~/Data/CityStreet/Street_groundplane_pmap.h5'), 'r') as f:
    #     pmap = np.asarray(f['v_pmap_GP'])
    # og_gt = dataset_train.prepare_cropped_gt(pmap, hw_random, frame=636)
    # 对于每个patch和各个视角下的mask

    # dataloader = DataLoader(dataset_train, 1, False, num_workers=4)
    # imgs, imgs_gt, detec_map_gt, count_map_gt, view_mask, hw_random, frame = next(iter(dataloader))
