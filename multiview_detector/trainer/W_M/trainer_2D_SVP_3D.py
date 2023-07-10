import time
import torch
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.utils.nms import nms
from multiview_detector.utils.meters import AverageMeter
from multiview_detector.utils.image_utils import add_heatmap_to_image
from multiview_detector.utils.gaussian_mse import target_transform
from multiview_detector.utils.person_help import *


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer_2D_SVP_3D(BaseTrainer):
    def __init__(self, model, logdir, denormalize, **kwargs):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.logdir = logdir
        self.denormalize = denormalize

        self.weight_svp = kwargs['weight_svp']
        self.weight_2D = kwargs['weight_2D']
        self.fix_2D = kwargs['fix_2D']
        self.fix_svp = kwargs['fix_svp']
        self.cls_thres = kwargs['cls_thres']
        self.nms_thres = kwargs['nms_thres']
        self.dist_thres = kwargs['dist_thres']
        self.current_dir = self.logdir + '/alljpgs/current_epoch'
        os.makedirs(self.current_dir, exist_ok=True)

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        for batch_idx, (data, map_gt, imgs_gt, _) in enumerate(data_loader):
            # imgs_gt = torch.cat(imgs_gt)
            optimizer.zero_grad()
            imgs_res, view_mask, view_gp_res, map_res, weight_norm = self.model(data)
            t_f = time.time()
            t_forward += t_f - t_b

            cuda_device = imgs_res.device
            # 2D
            gaussian_img_gt = target_transform(imgs_res,
                                               imgs_gt[0], data_loader.dataset.img_kernel)
            loss_2D = F.mse_loss(imgs_res, gaussian_img_gt.to(imgs_res.device))
            # loss_2D = F.mse_loss(imgs_res, imgs_gt[0].to(cuda_device))

            # SVP
            view_masked_map_gt = map_gt.to(view_mask.device) * view_mask
            view_gp_gt = target_transform(view_gp_res, view_masked_map_gt, data_loader.dataset.map_kernel)
            loss_SVP = F.mse_loss(view_gp_res, view_gp_gt.to(view_gp_res.device))

            # loss 3D
            gaussian_map_gt = target_transform(map_res, map_gt, data_loader.dataset.map_kernel)
            loss_3D = F.mse_loss(map_res, gaussian_map_gt.to(map_res.device)).to(cuda_device)
            loss = loss_3D + (self.weight_svp * loss_SVP + self.weight_2D * loss_2D) / data_loader.dataset.num_cam
            losses += loss.item()

            loss.backward()
            optimizer.step()
            losses += loss.item()

            t_b = time.time()
            t_backward += t_b - t_f

            t_b = time.time()
            t_backward += t_b - t_f
            if (batch_idx + 1) % log_interval == 0:
                t1 = time.time()
                t_epoch = t1 - t0
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                print(f'Train Epoch: {epoch}, Batch:{batch_idx + 1}, Loss: {losses / (batch_idx + 1):.6f}, '
                      f'loss_2D:{loss_2D:.6f}, loss_SVP:{loss_SVP:.6f}, loss_3D:{loss_3D:.6f}, '
                      f'lr:{current_lr:.8f}， '
                      f'Time: {t_epoch:.1f} (f{t_forward / batch_idx:.3f}+b{t_backward / batch_idx:.3f}),'
                      f' maxima_viewgp:{view_gp_res.max():.3f}, maxima_GP:{map_res.max():.3f}')
            if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                cyclic_scheduler.step()
        if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.LambdaLR):
            cyclic_scheduler.step()  # For Lambda
        t1 = time.time()
        t_epoch = t1 - t0
        losses /= len(data_loader)
        print(f'Train Epoch: {epoch}, Batch:{len(data_loader)}, Loss: {losses:.6f}, Time: {t_epoch:.3f} ')

        return losses

    def test(self, data_loader, epoch_resdir, res_fpath=None, gt_fpath=None, visualize=True):
        print('Testing...')
        self.model.eval()
        losses = 0
        batch_mae_2D = batch_mae_svp = 0
        all_res_list = []
        t0 = time.time()
        if res_fpath is not None:
            assert gt_fpath is not None
        for batch_idx, (data, dmap_gt, imgs_gt, frame) in enumerate(data_loader):
            # imgs_gt = torch.cat(imgs_gt)
            with torch.no_grad():
                imgs_res, view_gp_res, map_res, views_mask, weight_norm = self.model(data)
            if res_fpath is not None:
                map_grid_res = map_res.detach().cpu().squeeze()
                v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)
                grid_ij = (map_grid_res > self.cls_thres).nonzero()
                if data_loader.dataset.base.indexing == 'xy':
                    grid_xy = grid_ij[:, [1, 0]]
                else:
                    grid_xy = grid_ij
                all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                               data_loader.dataset.grid_reduce, v_s], dim=1))

            cuda_device = imgs_res.device
            # 2D loss
            loss_2D = F.mse_loss(imgs_res, imgs_gt.to(cuda_device))
            mae_2D = abs(imgs_res.sum() - imgs_gt.sum())
            # SVP loss
            view_gp_gt = dmap_gt.to(view_gp_res.device) * views_mask.to(cuda_device)
            loss_SVP = F.mse_loss(view_gp_res, view_gp_gt).to(cuda_device)
            # loss 3D
            loss_3D = F.mse_loss(map_res, dmap_gt.to(map_res.device)).to(cuda_device)

            loss_2D /= data_loader.dataset.num_cam
            loss_SVP /= data_loader.dataset.num_cam
            mae_2D /= data_loader.dataset.num_cam

            if self.fix_2D == 1 and self.fix_svp == 0:
                # print(f'loss=loss_3D + weight_svp * loss_SVP')
                loss = loss_3D + self.weight_svp * loss_SVP
            elif self.fix_2D == 1 and self.fix_svp == 1:
                loss = loss_3D
            elif self.fix_2D == 0 and self.fix_svp == 0:
                # print(f'loss=loss_3D + weight_svp * loss_SVP + wegiht_2D * loss_2D')
                loss = loss_3D + self.weight_svp * loss_SVP + self.weight_2D * loss_2D
            else:
                loss = 0
            losses += loss.item()

            if visualize and epoch_resdir and batch_idx % 100 == 0:
                for view in range(data_loader.dataset.num_cam):
                    # visualizing the heatmap for per-view estimation
                    heatmap0_head = imgs_res[view:view + 1][0, 0].detach().cpu().numpy().squeeze()
                    # heatmap0_foot = imgs_res[0][0, 1].detach().cpu().numpy().squeeze()
                    img0 = self.denormalize(data[0, view]).cpu().numpy().squeeze().transpose([1, 2, 0])
                    img0 = Image.fromarray((img0 * 255).astype('uint8'))
                    head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
                    head_cam_result.save(os.path.join(epoch_resdir, f'frame{str(frame.item())}_cam{view}_head.jpg'))

                    # visualization of single view prediction and its gt
                    fig = plt.figure()
                    subplt0 = fig.add_subplot(211, title="view_target")
                    subplt1 = fig.add_subplot(212, title='view_output')

                    subplt0.imshow((dmap_gt * views_mask[view:view + 1].to(dmap_gt.device)).cpu().squeeze())
                    subplt1.imshow(view_gp_res[view].cpu().squeeze().numpy())
                    plt.savefig(os.path.join(epoch_resdir, f'SVP_frame{str(frame.item())}_view{view}.jpg'))
                    plt.close(fig)

                    # weight map
                    plt.imshow(weight_norm[view].detach().squeeze().cpu())
                    # plt.colorbar()
                    plt.savefig(os.path.join(epoch_resdir, f'Norm_weight_frame{str(frame.item())}_view{view + 1}.jpg'))
            if visualize and epoch_resdir and batch_idx % 10 == 0:
                fig = plt.figure()
                subplt0 = fig.add_subplot(211, title="map_gt")
                subplt1 = fig.add_subplot(212, title='map_res')
                subplt0.imshow(dmap_gt.cpu().squeeze())
                subplt1.imshow(map_res.cpu().squeeze().numpy())
                plt.savefig(os.path.join(epoch_resdir, f'Final_prediction_frame{str(frame.item())}.jpg'))
                plt.close(fig)

        recall, precision, moda, modp, f1_score = self.metric_cal(res_fpath, gt_fpath, all_res_list, data_loader,
                                                                  self.nms_thres, self.dist_thres)
        print(f'ct={self.cls_thres}, nt={self.nms_thres}, dt={self.dist_thres}, '
              f'moda: {moda:.1f}%, modp: {modp:.1f}%, precision: {precision:.1f}%, recall: {recall:.1f}%, F1_score:{f1_score:.2f}')
        # base_nt = 20
        # base_dt = 20
        # recall_base, precision_base, moda_base, modp_base, f1_score_base = self.metric_cal(res_fpath, gt_fpath,
        #                                                                                    all_res_list, data_loader,
        #                                                                                    nt=base_nt, dt=base_dt)
        # print(
        #     f'ct={self.cls_thres}, nt={base_nt}, dt={base_dt}, moda_base: {moda_base:.1f}%, modp_base: {modp_base:.1f}%, precision_base: {precision_base:.1f}%, '
        #     f'recall_base: {recall_base:.1f}%, f1_score_base:{f1_score_base:.2f}')
        t1 = time.time()
        batch_mae_2D /= len(data_loader)
        losses /= len(data_loader)
        print(
            f'Test time:{t1 - t0:.1f}, losses:{losses:.6f}, batch_mae_2D={batch_mae_2D:.1f}, batch_mae_svp={batch_mae_svp:.1f}')
        return losses, moda, f1_score

    def metric_cal(self, res_fpath, gt_fpath, all_res_list, data_loader, nt, dt):
        moda = f1_score = 0
        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, nt, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')

            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                     dt, data_loader.dataset.base.__name__)

            # If you want to use the unofiicial python evaluation tool for convenient purposes.
            # recall, precision, modp, moda = python_eval(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
            #                                             data_loader.dataset.base.__name__)
            f1_score = 2 * precision * recall / (precision + recall + 1e-12)
            return recall, precision, moda, modp, f1_score

    def cls_test(self, data_loader, epoch_resdir=None, res_fpath=None, gt_fpath=None, visualize=True, first_test=False):
        cls_thres_list = [0.4, 0.6, 0.8]
        losses = 0
        all_cls_res_list = [[] for _ in range(len(cls_thres_list))]
        t0 = time.time()
        if res_fpath is not None:
            assert gt_fpath is not None
        for batch_idx, (data, map_gt, imgs_gt, frame) in enumerate(data_loader):
            with torch.no_grad():
                imgs_res, view_mask, view_gp_res, map_res, weight_norm = self.model(data)
            cuda_device = imgs_res.device
            # 2D
            gaussian_img_gt = target_transform(imgs_res, imgs_gt[0], data_loader.dataset.img_kernel)
            loss_2D = F.mse_loss(imgs_res, gaussian_img_gt.to(imgs_res.device))
            # loss_2D = F.mse_loss(imgs_res, imgs_gt[0].to(cuda_device))

            # SVP
            view_masked_map_gt = map_gt.to(view_mask.device) * view_mask
            view_gp_gt = target_transform(view_gp_res, view_masked_map_gt, data_loader.dataset.map_kernel)
            loss_SVP = F.mse_loss(view_gp_res, view_gp_gt.to(view_gp_res.device))

            # loss 3D
            gaussian_map_gt = target_transform(map_res, map_gt, data_loader.dataset.map_kernel)
            loss_3D = F.mse_loss(map_res, gaussian_map_gt.to(map_res.device)).to(cuda_device)
            loss = loss_3D + (self.weight_svp * loss_SVP + self.weight_2D * loss_2D) / data_loader.dataset.num_cam
            losses += loss.item()

            if visualize and batch_idx % 20 == 0:
                if epoch_resdir is None:
                    epoch_resdir = self.current_dir
                for view in range(0, data_loader.dataset.num_cam):
                    if self.fix_2D == 0 or first_test:
                        # visualizing the heatmap for per-view estimation
                        heatmap0_head = imgs_res[view, 0].detach().cpu().numpy().squeeze()
                        heatmap0_foot = imgs_res[view, 1].detach().cpu().numpy().squeeze()
                        img0 = self.denormalize(data[0, view]).cpu().numpy().squeeze().transpose([1, 2, 0])
                        img0 = Image.fromarray((img0 * 255).astype('uint8'))
                        head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
                        head_cam_result.save(os.path.join(epoch_resdir, f'frame{frame.item()}_cam{view}_head.jpg'))
                        foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
                        foot_cam_result.save(os.path.join(epoch_resdir, f'frame{frame.item()}_cam{view}_foot.jpg'))

                    # visualization of single view prediction and its gt
                    if self.fix_svp == 0 or first_test:
                        fig = plt.figure()
                        subplt0 = fig.add_subplot(211, title="view_target")
                        subplt1 = fig.add_subplot(212, title='view_output')
                        subplt0.imshow(view_gp_gt[view].cpu().squeeze())
                        subplt1.imshow(view_gp_res[view].cpu().squeeze().numpy())
                        plt.savefig(os.path.join(epoch_resdir, f'SVP_frame{str(frame.item())}_view{view}.jpg'))
                        plt.close(fig)
                        # weight map
                        plt.imshow(weight_norm[view].detach().squeeze().cpu())
                        # plt.colorbar()
                        plt.savefig(
                            os.path.join(epoch_resdir, f'Norm_weight_frame{str(frame.item())}_view{view + 1}.jpg'))

            if visualize and epoch_resdir and batch_idx % 5 == 0:
                fig = plt.figure()
                subplt0 = fig.add_subplot(211, title="map_gt")
                subplt1 = fig.add_subplot(212, title='map_res')
                subplt0.imshow(gaussian_map_gt[0].cpu().squeeze())
                subplt1.imshow(map_res.cpu().squeeze().numpy())
                plt.savefig(os.path.join(epoch_resdir, f'Final_prediction_frame{str(frame.item())}.jpg'))
                plt.close(fig)
            fig = plt.figure()
            subplt0 = fig.add_subplot(211, title="map_gt")
            subplt1 = fig.add_subplot(212, title='map_res')
            subplt0.imshow(gaussian_map_gt[0].cpu().squeeze())
            subplt1.imshow(map_res.cpu().squeeze().numpy())
            plt.savefig(os.path.join(epoch_resdir, f'Comparison.jpg'))
            plt.close(fig)

            for i, cls_thres in enumerate(cls_thres_list):
                if res_fpath is not None:
                    map_grid_res = map_res.detach().cpu().squeeze()
                    v_s = map_grid_res[map_grid_res > cls_thres].unsqueeze(1)
                    grid_ij = (map_grid_res > cls_thres).nonzero()
                    if data_loader.dataset.base.indexing == 'xy':
                        grid_xy = grid_ij[:, [1, 0]]
                    else:
                        grid_xy = grid_ij
                    all_cls_res_list[i].append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                                          data_loader.dataset.grid_reduce, v_s], dim=1))
        t1 = time.time()
        print(f'Test all batches consume time:{t1 - t0:.2f}')
        MODA = F1_SCORE = []
        if res_fpath is not None:
            for cls_thres, all_res_list in zip(cls_thres_list, all_cls_res_list):
                all_res_list = torch.cat(all_res_list, dim=0)
                np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + f'/all_res_cls{cls_thres}.txt',
                           all_res_list.numpy(), '%.8f')
                res_list = []
                for frame in np.unique(all_res_list[:, 0]):
                    res = all_res_list[all_res_list[:, 0] == frame, :]
                    positions, scores = res[:, 1:3], res[:, 3]
                    ids, count = nms(positions, scores, self.nms_thres, np.inf)
                    res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
                res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
                np.savetxt(res_fpath, res_list, '%d')
                recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                         self.dist_thres, data_loader.dataset.base.__name__)
                f1_score = 2 * recall * precision / (recall + precision + 1e-12)
                MODA.append(moda)
                F1_SCORE.append(f1_score)
                print(f'ct={cls_thres},nt={self.nms_thres} , dt={self.dist_thres}, '
                      f'moda: {moda:.1f}%, modp: {modp:.1f}%, precision: {precision:.1f}%, '
                      f'recall: {recall:.1f}% , F1_score:{f1_score:.1f}')
        t2 = time.time()
        print(f'Test 3 cls_thres cost time:{t2 - t1:.1f}')
        return losses / len(data_loader)
