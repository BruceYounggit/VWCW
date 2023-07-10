import json
import os
from math import sqrt, pow
import sys

import h5py
import kornia
import matplotlib.pyplot as plt
import numpy as np
import torch
from multiview_detector.utils.logger import Logger
from multiview_detector.models.CityStreet.spatial_transformer import SpatialTransformer_v3
from multiview_detector.utils.person_help import *


def get_imgcoord2worldgrid_matrices(num_cam, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
    projection_matrices = {}
    for cam in range(num_cam):
        worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)

        worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
        imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
        # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
        # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
        permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
        # projection_matrices[cam]=imgcoord2worldgrid_mat
        pass
    return projection_matrices


def wildtrack_multiviewx_prepare_gp_gt(base, gt_fpath, num_cam):
    og_gt = []
    for fname in sorted(os.listdir(os.path.join(base.root, 'annotations_positions'))):
        frame = int(fname.split('.')[0])
        with open(os.path.join(base.root, 'annotations_positions', fname)) as json_file:
            all_pedestrians = json.load(json_file)
        for single_pedestrian in all_pedestrians:
            def is_in_cam(cam):
                return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                            single_pedestrian['views'][cam]['xmax'] == -1 and
                            single_pedestrian['views'][cam]['ymin'] == -1 and
                            single_pedestrian['views'][cam]['ymax'] == -1)

            in_cam_range = sum(is_in_cam(cam) for cam in range(num_cam))
            if not in_cam_range:
                continue
            grid_x, grid_y = base.get_worldgrid_from_pos(single_pedestrian['positionID'])
            og_gt.append(np.array([frame, grid_x, grid_y]))
    og_gt = np.stack(og_gt, axis=0)
    os.makedirs(os.path.dirname(gt_fpath), exist_ok=True)
    np.savetxt(gt_fpath, og_gt, '%d')
    # return gt


def wildtrack():
    from multiview_detector.datasets.W_M.Wildtrack import Wildtrack
    sys.stdout = Logger(os.path.join('/home/yunfei/Study/MVD_VCW/Otherlogs', 'wildtrack_projection_error.txt'))
    base = Wildtrack(root='/home/yunfei/Data/Wildtrack')
    # img_shape = [1080, 1920]
    reducedgrid_shape = (120, 360)
    imgcoord2worldgrid_matrices = get_imgcoord2worldgrid_matrices(base.num_cam,
                                                                  base.intrinsic_matrices,
                                                                  base.extrinsic_matrices,
                                                                  base.worldgrid2worldcoord_mat)
    upsample_shape = list(map(lambda x: int(x / 4), base.img_shape))
    img_reduce = np.array(base.img_shape) / np.array(upsample_shape)
    # img_reduce = 4
    grid_reduce = 4
    img_zoom_mat = np.diag(np.append(img_reduce, [1]))
    map_zoom_mat = np.diag(np.append(np.ones([2]) / grid_reduce, [1]))
    proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                 for cam in range(base.num_cam)]
    # proj_test
    img_mat = torch.ones(1, 1, *upsample_shape)
    # for cam in range(base.num_cam):
    #     world_mat = kornia.geometry.warp_perspective(img_mat, proj_mats[cam].float()[None], reducedgrid_shape)
    #     plt.imshow(world_mat.squeeze())
    #     plt.show()
    frame_count = 0
    total_head_error = total_foot_error = 0
    for fname in sorted(os.listdir(os.path.join(base.root, 'annotations_positions'))):
        frame = int(fname.split('.')[0])
        frame_count += 1
        with open(os.path.join(base.root, 'annotations_positions', fname)) as json_file:
            all_pedestrians = json.load(json_file)
        frame_head_error = frame_foot_error = 0
        people_count = 0
        for single_pedestrian in all_pedestrians:
            def is_in_cam(cam):
                return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                            single_pedestrian['views'][cam]['xmax'] == -1 and
                            single_pedestrian['views'][cam]['ymin'] == -1 and
                            single_pedestrian['views'][cam]['ymax'] == -1)

            in_cam_range = sum(is_in_cam(cam) for cam in range(base.num_cam))
            if not in_cam_range:
                continue
            else:
                people_count += 1
            # 世界网格坐标
            grid_x, grid_y = base.get_worldgrid_from_pos(single_pedestrian['positionID'])
            real_coord = (int(grid_x / 4), int(grid_y / 4))
            # 图像坐标
            cam_head_error = cam_foot_error = 0.
            cam_head_count = cam_foot_count = 0
            for cam in range(base.num_cam):
                x = max(min(int((single_pedestrian['views'][cam]['xmin'] +
                                 single_pedestrian['views'][cam]['xmax']) / 2), base.img_shape[1] - 1), 0)
                y_head = max(single_pedestrian['views'][cam]['ymin'], 0)
                y_foot = min(single_pedestrian['views'][cam]['ymax'], base.img_shape[0] - 1)
                x = int(x / 4)
                y_foot = int(y_foot / 4)
                y_head = int(y_head / 4)
                img_coord_head = (y_head, x)
                img_coord_foot = (y_foot, x)
                world_coord_head = proj_solve(img_coord_head, upsample_shape, reducedgrid_shape, proj_mats[cam])
                world_coord_foot = proj_solve(img_coord_foot, upsample_shape, reducedgrid_shape, proj_mats[cam])
                if world_coord_head is not None:
                    cam_head_error += Eucliden_error(world_coord_head, real_coord)
                    cam_head_count += 1
                if world_coord_foot is not None:
                    cam_foot_error += Eucliden_error(world_coord_foot, real_coord)
                    cam_foot_count += 1
            # 视角平均
            cam_head_error /= (cam_head_count + 1e-18)
            cam_foot_error /= (cam_foot_count + 1e-18)

            frame_head_error += cam_head_error
            frame_foot_error += cam_foot_error
        frame_foot_error /= people_count
        frame_head_error /= people_count
        total_foot_error += frame_foot_error
        total_head_error += frame_head_error
        print(f'Index:{frame_count}, frame={frame}, {people_count}people contained:\n'
              f'frame_head_error:{total_head_error / frame_count:.3f}, frame_foot_error:{total_foot_error / frame_count:.3f}')

def multiviewx():
    from multiview_detector.datasets.W_M.MultiviewX import MultiviewX
    # sys.stdout = Logger(os.path.join('/home/yunfei/Study/MVD_VCW/Otherlogs', 'multiviewx_projection_error.txt'))
    base = MultiviewX(root='/home/yunfei/Data/MultiviewX')
    # img_shape = [1080, 1920]
    reducedgrid_shape = (160, 250)
    imgcoord2worldgrid_matrices = get_imgcoord2worldgrid_matrices(base.num_cam,
                                                                  base.intrinsic_matrices,
                                                                  base.extrinsic_matrices,
                                                                  base.worldgrid2worldcoord_mat)
    upsample_shape = list(map(lambda x: int(x / 4), base.img_shape))
    img_reduce = np.array(base.img_shape) / np.array(upsample_shape)
    # img_reduce = 4
    grid_reduce = 4
    img_zoom_mat = np.diag(np.append(img_reduce, [1]))
    map_zoom_mat = np.diag(np.append(np.ones([2]) / grid_reduce, [1]))
    proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                 for cam in range(base.num_cam)]
    # proj_test
    # img_mat = torch.ones(1, 1, *upsample_shape)
    # for cam in range(base.num_cam):
    #     world_mat = kornia.geometry.warp_perspective(img_mat, proj_mats[cam].float()[None], reducedgrid_shape)
    #     plt.imshow(world_mat.squeeze())
    #     plt.show()
    frame_count = 0
    total_head_error = total_foot_error = 0
    for fname in sorted(os.listdir(os.path.join(base.root, 'annotations_positions'))):
        frame = int(fname.split('.')[0])
        frame_count += 1
        with open(os.path.join(base.root, 'annotations_positions', fname)) as json_file:
            all_pedestrians = json.load(json_file)
        frame_head_error = frame_foot_error = 0
        people_count = 0
        for single_pedestrian in all_pedestrians:
            def is_in_cam(cam):
                return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                            single_pedestrian['views'][cam]['xmax'] == -1 and
                            single_pedestrian['views'][cam]['ymin'] == -1 and
                            single_pedestrian['views'][cam]['ymax'] == -1)

            in_cam_range = sum(is_in_cam(cam) for cam in range(base.num_cam))
            if not in_cam_range:
                continue
            else:
                people_count += 1
            # 世界网格坐标
            grid_x, grid_y = base.get_worldgrid_from_pos(single_pedestrian['positionID'])
            # real_coord = (int(grid_x / 4), int(grid_y / 4)) 与wildtrack相反
            real_coord = (int(grid_y / 4), int(grid_x / 4))
            # 图像坐标
            cam_head_error = cam_foot_error = 0.
            cam_head_count = cam_foot_count = 0
            for cam in range(base.num_cam):
                x = max(min(int((single_pedestrian['views'][cam]['xmin'] +
                                 single_pedestrian['views'][cam]['xmax']) / 2), base.img_shape[1] - 1), 0)
                y_head = max(single_pedestrian['views'][cam]['ymin'], 0)
                y_foot = min(single_pedestrian['views'][cam]['ymax'], base.img_shape[0] - 1)
                x = int(x / 4)
                y_foot = int(y_foot / 4)
                y_head = int(y_head / 4)
                img_coord_head = (y_head, x)
                img_coord_foot = (y_foot, x)
                world_coord_head = proj_solve(img_coord_head, upsample_shape, reducedgrid_shape, proj_mats[cam])
                world_coord_foot = proj_solve(img_coord_foot, upsample_shape, reducedgrid_shape, proj_mats[cam])
                if world_coord_head is not None:
                    cam_head_error += Eucliden_error(world_coord_head, real_coord)
                    cam_head_count += 1
                if world_coord_foot is not None:
                    cam_foot_error += Eucliden_error(world_coord_foot, real_coord)
                    cam_foot_count += 1
            # 视角平均
            cam_head_error /= (cam_head_count + 1e-18)
            cam_foot_error /= (cam_foot_count + 1e-18)

            frame_head_error += cam_head_error
            frame_foot_error += cam_foot_error
        frame_foot_error /= people_count
        frame_head_error /= people_count
        total_foot_error += frame_foot_error
        total_head_error += frame_head_error
        print(f'Index:{frame_count}, frame={frame}, {people_count}people contained:\n'
              f'frame_head_error:{total_head_error / frame_count:.3f}, frame_foot_error:{total_foot_error / frame_count:.3f}')

def Eucliden_error(x, y):
    return sqrt(pow(x[0] - y[0], 2) + pow(x[1] - y[1], 2))


def find_coord(tensor_mat):
    if tensor_mat.max() <= 0:
        return None
    else:
        return (tensor_mat == tensor_mat.max()).nonzero()[0][-2:]


def proj_solve(img_coord, upsample_shape, reducedgrid_shape, proj_mat):
    proj_mat = proj_mat.float()[None]
    img_mat = torch.zeros(1, 1, *upsample_shape)

    img_mat[0][0][img_coord[0]][img_coord[1]] = 10

    world_mat = kornia.geometry.warp_perspective(img_mat, proj_mat, reducedgrid_shape)

    world_coord = find_coord(world_mat)
    return world_coord


if __name__ == '__main__':
    # wildtrack()
    multiviewx()
