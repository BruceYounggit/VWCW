import os
import sys
import h5py
import numpy as np
from math import sqrt, pow
from multiview_detector.utils.logger import Logger


def prepare_gt(gt_fpath):
    og_gt = []
    with h5py.File(os.path.expanduser('~/Data/CityStreet/Street_groundplane_pmap.h5'), 'r') as f:
        for i in range(f['v_pmap_GP'].shape[0]):
            singlePerson_Underframe = f['v_pmap_GP'][i]
            frame = int(singlePerson_Underframe[0])
            # personID = int(singlePerson_Underframe[1])
            # 原论文grid_x 为H这条边，singleFrame_underframe[3]指的是cy，最大值不超过768
            # 原论文grid_y 为W这条边，singleFrame_underframe[2]指的是cx，最大值不超过640
            grid_y = int(singlePerson_Underframe[2]) * 4  # 乘以4之后，每个pixel代表2.5cm
            grid_x = int(singlePerson_Underframe[3]) * 4  # [3072, 2560]
            # height = int(singlePerson_Underframe[4])
            og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
    os.makedirs(os.path.dirname(gt_fpath), exist_ok=True)
    np.savetxt(gt_fpath, og_gt, '%d')
    f.close()


if __name__ == '__main__':
    method = ['Ours', 'MVDet', 'SHOT', 'MVDeTr', '3D ROM']
    gt_fpath = os.path.join('/home/yunfei/Study/MVD_VCW/Otherlogs', 'gt.txt')
    if not os.path.exists(gt_fpath):
        prepare_gt(gt_fpath)
    res_path = ['/home/yunfei/Study/Baseline_MVDet/logs/citystreet_frame/2D_SVP_3D/2022-11-17_09-56-39_iccv/test.txt',
                '/home/yunfei/Study/Baseline_MVDet/logs/citystreet_frame/2D_3D/2022-11-16_02-01-49_max_BASE/test.txt',
                '/home/yunfei/Study/SHOT_Citystreet/logs/citystreet_frame/default/2022-08-26_16-56-40_selected/test.txt',
                '/home/yunfei/Study/CityStreet_MVDeTr/logs/citystreet/2022-10-28_17-17-49_pretrainT_lr0.0005_baseR0.1_neck128_out0_alpha1.0_worldRK2_6_imgRK1--ICCV/test.txt',
                '/home/yunfei/Study/3DROM_city/logs/citystreet_frame/default/2022-11-02_19-08-37-[1600,1700,1800,1900]/test.txt']
    gt = np.loadtxt(fname=gt_fpath)
    for i in range(len(res_path)):
        sys.stdout = Logger(
            os.path.join('/home/yunfei/Study/MVD_VCW/Otherlogs', f'iccv_table2_{method[i]}_counting_results.txt'))
        # prepare_gt(gt_fpath)

        res = np.loadtxt(res_path[i])
        print(f'gt_fpath:{gt_fpath}, res_path:{res_path[i]}')
        MAE = MSE = 0.
        frame_range = range(1236, 1636, 2)
        for frame in frame_range:
            frame_gt = gt[gt[:, 0] == frame, :]
            frame_res = res[res[:, 0] == frame, :]
            count_gt = frame_gt.shape[0]
            count_res = frame_res.shape[0]
            frame_MAE = abs(count_res - count_gt)
            frame_MSE = pow(count_res - count_gt, 2)
            print(f'frame={frame}, mae={frame_MAE}, mse={frame_MSE}.')
            MAE += frame_MAE
            MSE += frame_MSE
        MAE /= len(frame_range)
        MSE /= len(frame_range)
        RMSE = sqrt(MSE)
        print(f'Average metrics under {len(frame_range)} test frames.\n'
              f'MAE={MAE:.3f}. MSE={MSE:.3f}, RMSE={RMSE:.3f}')
