import json
import os
from math import sqrt, pow
import sys

import h5py
import numpy as np
import torch
import multiview_detector.datasets.CityStreet.camera_proj_Zhang as proj
from multiview_detector.utils.logger import Logger
from multiview_detector.models.CityStreet.spatial_transformer import SpatialTransformer_v3
from multiview_detector.utils.person_help import *


def dist_error(proj_coords, gp_coord):
    view_error = {0: 0, 1: 0, 2: 0}
    for view in range(3):
        if proj_coords[view] is not None:
            view_error[view] += sqrt(
                pow(proj_coords[view][0] - gp_coord[0], 2) + pow(proj_coords[view][1] - gp_coord[1], 2))
    return view_error


def find_coord(input_tensor: torch.Tensor, real_coord):
    proj_coords = {}
    input_tensor = input_tensor.detach().cpu()
    # min_dis = 1e8
    # cur_dis =  1e-
    for view in range(3):
        if input_tensor[view].max() > 0:
            # 找出一系列点中距离目标点最近的点坐标
            # res = (input_tensor == input_tensor[view].max()).nonzero()  # 所有大于0的点
            # for i in range(res.shape[0]):
            res = (input_tensor[view][0] > 0).nonzero()
            mindist = 1e8
            for i in range(res.shape[0]):
                if mindist > sqrt(pow(res[i][0] - real_coord[0], 2) + pow(res[i][1] - real_coord[1], 2)):
                    mindist = sqrt(pow(res[i][0] - real_coord[0], 2) + pow(res[i][1] - real_coord[1], 2))
                    proj_coords[view] = tuple(map(lambda x: float(x.item()), res[i]))
            # cur_dis += sqrt(pow(res[i][0] - real_coord[0], 2) + pow(res[i][1] - real_coord[1], 2))
            # min_dis = min(min_dis, cur_dis)
            # if min_dis == cur_dis
        else:
            proj_coords[view] = None
    return proj_coords


# def find_min_dist_coord(input_tensor, real_coord):

def world2image(view, worldcoords):
    N = worldcoords.shape[0]
    imgcoords = []
    for i in range(N):
        worldcoord = worldcoords[i, :]

        Xw = worldcoord[0].item()
        Yw = worldcoord[1].item()
        Zw = worldcoord[2].item()

        XYi = proj.World2Image(view, Xw, Yw, Zw)
        imgcoords.append(XYi)
    imgcoords = np.asarray(imgcoords)
    return imgcoords


def img2world(view, imgcoords):
    N = imgcoords.shape[0]
    worldcoods = []
    for i in range(N):
        imgcoord = imgcoords[i]
        id = imgcoord[0].item()
        Xi = imgcoord[1].item()
        Yi = imgcoord[2].item()
        Zi = imgcoord[3].item()

        XYw = proj.Image2World(view, Xi, Yi, Zi)
        worldcoods.append([id, *XYw])
    worldcoods = np.stack(worldcoods, axis=0)
    return worldcoods


def worldcoord2worldgrid(worldcoords, outshape=(768, 640)):
    grid_fac = 4 / (outshape[0] / 192)
    bbox = [352 * 0.8, 522 * 0.8]
    resolution_scaler = 76.25
    id = worldcoords[:, 0]
    # viewvec = np.ones_like(id) * view
    grid_rangeX = worldcoords[:, 1]
    grid_rangeY = worldcoords[:, 2]
    # grid_rangeX = (grid_rangeX * grid_fac - bbox[0]) * resolution_scaler
    # grid_rangeY = (grid_rangeY * grid_fac - bbox[1]) * resolution_scaler
    grid_rangeX = (grid_rangeX / resolution_scaler + bbox[0]) / grid_fac
    grid_rangeY = (grid_rangeY / resolution_scaler + bbox[1]) / grid_fac
    # 去除投影在【640，768】之外的点
    num = grid_rangeX.shape[0]
    for i in range(num):
        if grid_rangeX[i] < 0 or grid_rangeX[i] > outshape[0] or \
                grid_rangeY[i] < 0 or grid_rangeY[i] > outshape[1]:
            id[i] = -id[i]

    worldgrid = np.stack([id, grid_rangeX, grid_rangeY], axis=1)
    return worldgrid


def Eucliden_dist(x, y):
    return sqrt(pow(x[0] - y[0], 2) + pow(x[1] - y[1], 2))


def Vec_ndarray_dist(proj_grid, world_grid):
    # 计算当前投影网格所有点的误差
    world_grid_id_ls = list(world_grid[:, 0])
    Error = 0
    for i in range(proj_grid.shape[0]):
        if proj_grid[i][0] in world_grid_id_ls:
            proj_x = proj_grid[i][1]
            proj_y = proj_grid[i][2]
            current_world_grid = (world_grid[world_grid[:, 0] == proj_grid[i][0]]).squeeze()
            world_x = current_world_grid[1]
            world_y = current_world_grid[2]
            error = Eucliden_dist((proj_x, proj_y), (world_x, world_y))
            Error += error
    view_error = Error / sum(proj_grid[:, 0] > 0)
    return view_error, Error, sum(proj_grid[:, 0] > 0)


def citystreet(device, height):
    from multiview_detector.datasets.CityStreet.camera_proj_Zhang import Image2World, World2Image
    # sys.stdout = Logger(os.path.join('/home/yunfei/Study/MVD_VCW/Otherlogs',
    #                                  f'citystreet直接点投影_proj_height{height}_log.txt', ))
    # STN = SpatialTransformer_v3(input_size=[1, 380, 676, 1], output_size=(768, 640), device=device,
    #                             person_heights=[height])
    # ts_ones = torch.zeros(3, 1, 380, 676)
    #     # ts_ones[0][0][100][200] = 10
    #     # ts_ones[2][0][100][200] = 10
    #     # ts_ones[1][0][100][200] = 10
    #     # proj_ts = STN(ts_ones.permute(0, 2, 3, 1), height=1750).permute(0, 3, 1, 2)
    #     # pc = find_coord(proj_ts)
    # 直接进行点坐标的投影
    # imgcoords = np.random.randint(0, 380, size=(10, 2))
    # grid_z = np.ones(shape=(10,1))
    # imgcoords = np.concatenate([imgcoords, grid_z], axis=1)
    # world_coords = img2world('view1', imgcoords)

    json_files_dir = '/home/yunfei/Data/CityStreet/labels'
    data = {}

    # with open(os.path.join(json_files_dir, 'GP_pmap_height.json')) as file:
    #     data['GP'] = json.load(file)
    with open(os.path.join(json_files_dir, 'via_region_data_view1.json')) as file:
        data['view1'] = json.load(file)
    with open(os.path.join(json_files_dir, 'via_region_data_view2.json')) as file:
        data['view2'] = json.load(file)
    with open(os.path.join(json_files_dir, 'via_region_data_view3.json')) as file:
        data['view3'] = json.load(file)

    frame_range = range(636, 1406, 2)
    View_Error = {1: 0, 2: 0, 3: 0}
    T_error=T_count=0
    world_reduce=2
    with h5py.File('/home/yunfei/Data/CityStreet/Street_groundplane_pmap.h5', 'r') as f:
        for frame in frame_range:
            world_real_grid = []
            img_view_coords = {1: [], 2: [], 3: []}

            frame_part = f['v_pmap_GP'][f['v_pmap_GP'][:, 0] == frame, :]
            for i in range(frame_part.shape[0]):
                id = str(int(frame_part[i, 1]))
                real_x = frame_part[i, 2]/world_reduce
                real_y = frame_part[i, 3]/world_reduce
                # gp_region = data['GP']['frame_{:0>4}.jpg'.format(frame)]['regions']
                # 地面坐标
                # real_x = gp_region[id]['shape_attributes']['cx']
                # real_y = gp_region[id]['shape_attributes']['cy']
                # if real_x is None or real_y is None:
                #     continue
                # real_height = gp_region[id]['shape_attributes']['height']
                real_coord = (int(id), real_x, real_y)
                world_real_grid.append(real_coord)
                # stn_input = torch.zeros(3, 1, 380, 676).to(device)
                # id_occur = 0  # 该ID人出现在3个视角里次数
                for view in range(3):  # 该帧下所有视角内出现的人的图像坐标
                    view_region = data[f'view{view + 1}']['frame_{:0>4}.jpg'.format(frame)]['regions']
                    if id in view_region.keys():  # 地面人位于该视角内.
                        # 图像坐标
                        img_x = view_region[id]['shape_attributes']['cx']
                        img_y = view_region[id]['shape_attributes']['cy']
                        if img_x is not None and img_y is not None:
                            if 0 <= img_x < 2704 and 0 <= img_y < 1520:
                                img_view_coords[view + 1].append((int(id), img_x, img_y))

            world_real_grid = np.stack(world_real_grid, axis=0)
            index = (frame - 634) / 2
            print(f'frame={frame}:')
            for view in range(1, 4):
                img_view_coords[view] = np.stack(img_view_coords[view], axis=0)
                person_Z = np.ones(shape=(img_view_coords[view].shape[0], 1)) * height
                img_view_coords[view] = np.concatenate([img_view_coords[view], person_Z], axis=1)
                proj_coords = img2world(f'view{view}', img_view_coords[view])
                proj_coords_grid_view = worldcoord2worldgrid(proj_coords, outshape=(768/world_reduce, 640/world_reduce))
                view_error, total_error, total_points = Vec_ndarray_dist(proj_coords_grid_view, world_real_grid)
                View_Error[view] += view_error
                T_error+=total_error
                T_count+=total_points

                print(f'\tview{view}_error:{view_error:.3f}, avg_view{view}_error:{View_Error[view] / index:.3f}')
            # avg_point_error = (View_Error[1] + View_Error[2] + View_Error[3]) / (3.0 * index)
            print(f'avg_point_error={T_error/T_count:.3f}')


if __name__ == '__main__':
    # from multiview_detector.datasets.W_M.Wildtrack import Wildtrack
    # from multiview_detector.datasets.W_M.MultiviewX import MultiviewX
    device = 'cpu'
    citystreet(device, height=1750)
