import math
import os
import time

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from multiview_detector.evaluation.pyeval.evaluateDetection import evaluateDetection_py
from multiview_detector.utils.image_utils import add_heatmap_to_image
from multiview_detector.utils.nms import nms
from multiview_detector.utils.person_help import vis


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, logdir, denormalize, cls_thres=0.6, nms_thres=3, dist_thres=3, **kwargs):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.weight_2D = kwargs['weight_2D']
        self.weight_svp = kwargs['weight_svp']

        self.fix_2D = kwargs['fix_2D']
        self.fix_svp = kwargs['fix_svp']
        self.fix_weight = kwargs['fix_weight']

        self.cls_thres = cls_thres
        self.nms_thres = nms_thres
        self.dist_thres = dist_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.num_cam = model.num_cam
        self.current_dir = self.logdir + '/alljpgs'
        os.makedirs(self.current_dir, exist_ok=True)

    def train(self, data_loader, epoch, optimizer, log_interval=100, cyclic_scheduler=None, writer=None):
        self.model.train()
        losses = 0
        pf_losses = 0
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0

        # views_mask is used to cover gp_patch_views_dmap to produce loss_3D
        for batch_idx, (imgs, img_gt, masked_view_gp_gt, gp_gt, frame) in enumerate(data_loader):
            img_gt = img_gt.permute(1, 0, 2, 3)
            masked_view_gp_gt = masked_view_gp_gt.permute(1, 0, 2, 3)

            # optimizer.zero_grad()
            optimizer['optimizer1'].zero_grad()
            optimizer['optimizer2'].zero_grad()
            # img_res, view_gp_output = self.model(imgs)
            img_res, view_gp_output, gp_res, weight_mask = self.model(imgs)

            t_f = time.time()
            t_forward += t_f - t_b
            # loss_2D for RGB images
            loss_2D = F.mse_loss(img_res, img_gt.to(img_res.device))
            # loss_SVP for single-view prediction
            loss_svp = F.mse_loss(view_gp_output, masked_view_gp_gt.to(view_gp_output.device))
            loss_vcw = F.mse_loss(gp_res, gp_gt.to(gp_res.device))
            loss = loss_vcw + loss_svp * self.weight_svp + loss_2D * self.weight_2D
            loss.backward()
            optimizer['optimizer1'].step()
            optimizer['optimizer2'].step()

            # pf_losses += feat_fusion_loss.item() + pred_fusion_loss.item()
            losses += loss.item()
            t_b = time.time()
            t_backward += t_b - t_f
            if (batch_idx + 1) % log_interval == 0:
                t1 = time.time()
                t_epoch = t1 - t0
                current_lr = [optimizer['optimizer1'].state_dict()['param_groups'][0]['lr'],
                              optimizer['optimizer2'].state_dict()['param_groups'][0]['lr']]

                print(f'Train Epoch: {epoch}, Batch:{batch_idx + 1}, Loss: {losses / (batch_idx + 1):.6f}, '
                      f' loss_vcw:{loss_vcw:.6f}, loss_svp:{loss_svp:.6f}, loss_2D:{loss_2D:.6f}, '
                      f' lr1:{current_lr[0]:.8f},lr2:{current_lr[1]:.8f}, '
                      f'Time: {t_epoch:.1f} (f{t_forward / batch_idx:.3f}+b{t_backward / batch_idx:.3f}), '
                      f'view_gp_max:{view_gp_output.max():.3f}, joint_fusion_maxima:{gp_res.max():.3f}')

            if isinstance(cyclic_scheduler['scheduler1'], torch.optim.lr_scheduler.OneCycleLR):
                cyclic_scheduler['scheduler1'].step()
            if isinstance(cyclic_scheduler['scheduler2'], torch.optim.lr_scheduler.OneCycleLR):
                cyclic_scheduler['scheduler2'].step()

        if isinstance(cyclic_scheduler['scheduler1'], torch.optim.lr_scheduler.LambdaLR):
            cyclic_scheduler['scheduler1'].step()  # For Lambda
        if isinstance(cyclic_scheduler['scheduler2'], torch.optim.lr_scheduler.LambdaLR):
            cyclic_scheduler['scheduler2'].step()  # For Lambda

        t1 = time.time()
        t_epoch = t1 - t0
        losses /= len(data_loader)

        print(f'Train Epoch: {epoch}, Batch:{len(data_loader)}, Loss: {losses:.6f}, '
              f'Time: {t_epoch:.3f} ')
        return losses

    def test(self, data_loader, gt_fpath=None, visualize=False, epoch_resdir=None, first_test=False):
        self.model.eval()
        losses = 0
        t0 = time.time()
        all_res_list = []
        if epoch_resdir:
            res_fpath = os.path.join(epoch_resdir, 'res.txt')
        else:
            res_fpath = os.path.join(self.logdir, 'res.txt')
        for batch_idx, (imgs, img_gt, masked_view_gp_gt, gp_gt, frame) in enumerate(data_loader):
            frame = int(frame)
            img_gt = img_gt.permute(1, 0, 2, 3)
            masked_view_gp_gt = masked_view_gp_gt.permute(1, 0, 2, 3)
            with torch.no_grad():
                img_res, view_gp_output, gp_res, weight_mask = self.model(imgs)
                map_grid_res = gp_res.detach().cpu().squeeze()
                v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)
                grid_ij = (map_grid_res > self.cls_thres).nonzero()

                all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_ij.float() *
                                               data_loader.dataset.world_reduce, v_s], dim=1))
            # loss_2D for RGB images
            loss_2D = F.mse_loss(img_res, img_gt.to(img_res.device))
            # loss_SVP for single-view prediction
            loss_svp = F.mse_loss(view_gp_output, masked_view_gp_gt.to(view_gp_output.device))
            loss_vcw = F.mse_loss(gp_res, gp_gt.to(gp_res.device))
            loss = loss_vcw + loss_svp * self.weight_svp + loss_2D * self.weight_2D
            losses += loss
            # visualization
            if visualize and batch_idx % 100 == 0 and epoch_resdir:
                # if visualize:
                for view in range(0, data_loader.dataset.num_cam):
                    if self.fix_2D == 0 or first_test:
                        # visualizing the heatmap for per-view estimation
                        heatmap0_head = img_res[view, 0].detach().cpu().numpy().squeeze()
                        img0 = self.denormalize(imgs[0, view]).cpu().numpy().squeeze().transpose([1, 2, 0])
                        img0 = Image.fromarray((img0 * 255).astype('uint8'))
                        head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
                        head_cam_result.save(os.path.join(epoch_resdir, f'frame{frame}_cam{view}.jpg'))

                    # visualization of single view prediction and its gt
                    if self.fix_svp == 0 or first_test:
                        fig = plt.figure()
                        subplt0 = fig.add_subplot(121, title="view_target")
                        subplt1 = fig.add_subplot(122, title='view_output')
                        subplt0.imshow(masked_view_gp_gt[view].cpu().squeeze())
                        subplt1.imshow(view_gp_output[view].cpu().squeeze().numpy())
                        plt.savefig(os.path.join(epoch_resdir, f'SVP_frame{frame}_view{view}.jpg'))
                        plt.close(fig)

                    # w_mask vis
                    if self.fix_weight == 0 or first_test:
                        plt.imshow(weight_mask[view].detach().squeeze().cpu().numpy())
                        plt.colorbar()
                        plt.savefig(os.path.join(epoch_resdir, f'weight_mask_frame{frame}_view{view}.jpg'))
                        plt.close()

                # Fusion vis
                fig = plt.figure()
                subplt0 = fig.add_subplot(121, title="gp_target", xticks=[], yticks=[])
                subplt1 = fig.add_subplot(122, title='gp_res', xticks=[], yticks=[])
                subplt0.imshow(gp_gt.cpu().squeeze())
                subplt1.imshow(gp_res.detach().cpu().squeeze())
                # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=-0.4, hspace=0.2)
                plt.savefig(os.path.join(epoch_resdir, f'Fusion_res_frame{frame}.jpg'), dpi=600)
                plt.close(fig)

        t1 = time.time()
        moda = 0
        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, self.nms_thres, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')

            recall, precision, moda, modp = evaluateDetection_py(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                                 self.dist_thres,
                                                                 data_loader.dataset.base.__name__)
            F1_score = 2 * precision * recall / (precision + recall + 1e-12)
            print(f'moda: {moda:.1f}%, modp: {modp:.1f}%,'
                  f' precision: {precision:.1f}%, recall: {recall:.1f}%, F1_score:{F1_score:.1f}%')

        losses = losses / len(data_loader)
        print(f'Test time:{t1 - t0:.1f}, losses:{losses:.6f}')
        return losses

    def counting_test(self, data_loader, gt_fpath=None, visualize=False, epoch_resdir=None, first_test=False):
        self.model.eval()
        losses = 0
        t0 = time.time()
        MAE = 0
        MSE = 0
        all_res_list = []
        if epoch_resdir:
            res_fpath = os.path.join(epoch_resdir, 'res.txt')
        else:
            res_fpath = os.path.join(self.logdir, 'res.txt')
        for batch_idx, (imgs, img_gt, masked_view_gp_gt, gp_gt, frame) in enumerate(data_loader):
            frame = int(frame)
            img_gt = img_gt.permute(1, 0, 2, 3)
            masked_view_gp_gt = masked_view_gp_gt.permute(1, 0, 2, 3)

            with torch.no_grad():
                img_res, view_gp_output, gp_res, weight_mask = self.model(imgs)
                map_grid_res = gp_res.detach().cpu().squeeze()
                v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)
                grid_ij = (map_grid_res > self.cls_thres).nonzero()

                all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_ij.float() *
                                               data_loader.dataset.world_reduce, v_s], dim=1))
            # loss_2D for RGB images
            loss_2D = F.mse_loss(img_res, img_gt.to(img_res.device))
            # loss_SVP for single-view prediction
            loss_svp = F.mse_loss(view_gp_output, masked_view_gp_gt.to(view_gp_output.device))
            loss_vcw = F.mse_loss(gp_res, gp_gt.to(gp_res.device))
            mae = abs(gp_res.sum() / data_loader.dataset.facofmaxgt_gp - gp_gt.sum())
            mse = pow(gp_res.sum() / data_loader.dataset.facofmaxgt_gp - gp_gt.sum(), 2)
            MAE += mae
            MSE += mse
            if batch_idx%50==0:
                print(f'frame={frame},\n gpmae:{mae:.3f}, mse:{mse:.3f}')
                print(f'MAE: {MAE / (batch_idx + 1):.3f}, MSE:{MSE / (batch_idx + 1):.3f}')
            loss = loss_vcw + loss_svp * self.weight_svp + loss_2D * self.weight_2D
            losses += loss
            # visualization
            if visualize and batch_idx % 100 == 0 and epoch_resdir:
                # if visualize:
                for view in range(0, data_loader.dataset.num_cam):
                    if self.fix_2D == 0 or first_test:
                        # visualizing the heatmap for per-view estimation
                        heatmap0_head = img_res[view, 0].detach().cpu().numpy().squeeze()
                        img0 = self.denormalize(imgs[0, view]).cpu().numpy().squeeze().transpose([1, 2, 0])
                        img0 = Image.fromarray((img0 * 255).astype('uint8'))
                        head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
                        head_cam_result.save(os.path.join(epoch_resdir, f'frame{frame}_cam{view}.jpg'))

                    # visualization of single view prediction and its gt
                    if self.fix_svp == 0 or first_test:
                        fig = plt.figure()
                        subplt0 = fig.add_subplot(121, title="view_target")
                        subplt1 = fig.add_subplot(122, title='view_output')
                        subplt0.imshow(masked_view_gp_gt[view].cpu().squeeze())
                        subplt1.imshow(view_gp_output[view].cpu().squeeze().numpy())
                        plt.savefig(os.path.join(epoch_resdir, f'SVP_frame{frame}_view{view}.jpg'))
                        plt.close(fig)

                    # w_mask vis
                    if self.fix_weight == 0 or first_test:
                        plt.imshow(weight_mask[view].detach().squeeze().cpu().numpy())
                        plt.colorbar()
                        plt.savefig(os.path.join(epoch_resdir, f'weight_mask_frame{frame}_view{view}.jpg'))
                        plt.close()

                # Fusion vis
                fig = plt.figure()
                subplt0 = fig.add_subplot(121, title="gp_target", xticks=[], yticks=[])
                subplt1 = fig.add_subplot(122, title='gp_res', xticks=[], yticks=[])
                subplt0.imshow(gp_gt.cpu().squeeze())
                subplt1.imshow(gp_res.detach().cpu().squeeze())
                # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=-0.4, hspace=0.2)
                plt.savefig(os.path.join(epoch_resdir, f'Fusion_res_frame{frame}.jpg'), dpi=600)
                plt.close(fig)
        MAE /= len(data_loader)
        MSE /= len(data_loader)
        RMSE = math.sqrt(MSE)
        print(f'MSE: {MAE:.3f}, MSE:{MSE:.3f},RMSE: {RMSE:.3f}')

        t1 = time.time()
        moda = 0
        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, self.nms_thres, np.inf)
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')

            recall, precision, moda, modp = evaluateDetection_py(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                                                                 self.dist_thres,
                                                                 data_loader.dataset.base.__name__)
            F1_score = 2 * precision * recall / (precision + recall + 1e-12)
            print(f'moda: {moda:.1f}%, modp: {modp:.1f}%,'
                  f' precision: {precision:.1f}%, recall: {recall:.1f}%, F1_score:{F1_score:.1f}%')

        losses = losses / len(data_loader)
        print(f'Test time:{t1 - t0:.1f}, losses:{losses:.6f}')
        return losses
