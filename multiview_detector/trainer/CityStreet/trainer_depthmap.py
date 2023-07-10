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


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, logdir, denormalize, cls_thres=0.6, nms_thres=3, dist_thres=3, **kwargs):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.weight_2D = kwargs['weight_2D']
        self.weight_svp = kwargs['weight_svp']
        self.weight_ssv = kwargs['weight_ssv']
        self.fix_2D = kwargs['fix_2D']
        self.fix_svp = kwargs['fix_svp']

        self.cls_thres = cls_thres
        self.nms_thres = nms_thres
        self.dist_thres = dist_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.num_cam = model.num_cam
        self.logdir_imgpath = os.path.join(self.logdir, 'alljpgs')

        if not os.path.exists(self.logdir_imgpath):
            os.mkdir(self.logdir_imgpath)

    def train(self, data_loader, epoch, optimizer, log_interval=100, cyclic_scheduler=None, writer=None):
        self.model.train()
        losses = 0
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        Mae_3D_batches = 0
        # views_mask is used to cover gp_patch_views_dmap to produce loss_3D
        for batch_idx, (data, img_gt, detect_gt, _) in enumerate(data_loader):
            optimizer.zero_grad()
            # gp_map_res:[2,1,ph,pw], view_gp_map_res is same, w_mask:[p*v,ph,pw,1]
            detect_gt = detect_gt.unsqueeze(0)
            # gp_map_res, view_gp_map_res, w_mask = self.model(data, hw_random)
            img_res, view_gp_output, mean_out, w_mask, Dmap_consist_loss = self.model(data)
            t_f = time.time()
            t_forward += t_f - t_b
            # loss 2D
            loss_2D = F.mse_loss(img_res, img_gt.to(img_res.device))
            # loss_SVP
            loss_svp = F.mse_loss(view_gp_output, detect_gt.repeat(self.num_cam, 1, 1, 1).to(view_gp_output.device))
            # loss 3D
            loss_3D = F.mse_loss(mean_out, detect_gt.to(mean_out.device))
            mae_3D = abs(mean_out.sum() - detect_gt.sum()) / \
                     (data_loader.dataset.gaussian_kernel_sum * data_loader.dataset.facofmaxgt_gp)


            loss = loss_svp + loss_3D
            Mae_3D_batches += mae_3D
            loss.backward()
            optimizer.step()
            losses += loss.item()
            t_b = time.time()
            t_backward += t_b - t_f
            if (batch_idx + 1) % log_interval == 0:
                t1 = time.time()
                t_epoch = t1 - t0
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                print(f'Train Epoch: {epoch}, Batch:{batch_idx + 1}, Loss: {losses / (batch_idx + 1):.6f}, '
                      f'loss_3D:{loss_3D:.6f}, loss_svp:{loss_svp:.6f}, lr:{current_lr:.8f}, MAE_3D:{mae_3D:.3f}, '
                      f'Time: {t_epoch:.1f} (f{t_forward / batch_idx:.3f}+b{t_backward / batch_idx:.3f}), '
                      f'maxima:{gp_map_res.max():.+3f}')
            if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                cyclic_scheduler.step()
        if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.LambdaLR):
            cyclic_scheduler.step()  # For Lambda

        t1 = time.time()
        t_epoch = t1 - t0
        losses /= len(data_loader)
        Mae_3D_batches /= len(data_loader)
        print(f'Train Epoch: {epoch}, Batch:{len(data_loader)}, Loss: {losses:.6f}, '
              f'MAE_3D_batches:{Mae_3D_batches:.3f}, '
              f'Time: {t_epoch:.3f} ')

        return losses

    def test(self, data_loader, gt_fpath=None, visualize=False, epoch_resdir=None, writer=None):
        self.model.eval()
        losses = 0
        Mae_3D_batches = 0
        all_res_list = []
        t0 = time.time()
        res_fpath = os.path.join(epoch_resdir, 'res.txt')
        # og_gt = {p: [] for p in range(self.patch_num)}
        for batch_idx, (data, detect_gt, frame) in enumerate(data_loader):
            frame = int(frame)
            detect_gt = detect_gt.unsqueeze(0)
            with torch.no_grad():
                # gp_map_res, view_gp_map_res, w_mask = self.model(data, hw_random)
                # gp_map_res, w_mask = self.model(data)
                mean_out, gp_map_res, view_gp_out, w_mask = self.model(data)
                if res_fpath is not None:
                    map_grid_res = mean_out.detach().cpu().squeeze()
                    v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)
                    grid_ij = (map_grid_res > self.cls_thres).nonzero()

                    all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_ij.float() *
                                                   data_loader.dataset.world_reduce * 4, v_s], dim=1))
                # visualization for Ground plane

                if visualize and batch_idx % 100 == 0:
                    self.middle_visualize_comparation(mean_out.detach().cpu().squeeze(),
                                                      detect_gt.to(mean_out.device).cpu().squeeze(),
                                                      epoch_resdir,
                                                      title=f'Mean_out vs gt frame{frame}')
                    self.middle_visualize_comparation(detect_gt.cpu().squeeze(),
                                                      gp_map_res.detach().cpu().squeeze(),
                                                      epoch_resdir,
                                                      title=f'GT vs Feature_prediction frame{frame}')
                    for i in range(self.num_cam):
                        self.middle_visualize_comparation(view_gp_out[i].detach().cpu().squeeze(),
                                                          gp_map_res.detach().cpu().squeeze(),
                                                          epoch_resdir,
                                                          title=f'ViewOut vs Feature_prediction frame{frame}')

                    self.middle_visualize(w_mask.detach().cpu(), epoch_resdir, title=f'Weight_Map_frame{frame}')
            # loss 3D
            loss_3D = F.mse_loss(mean_out, detect_gt.to(mean_out.device))
            mae_3D = abs(mean_out.sum() - detect_gt.sum()) / \
                     (data_loader.dataset.gaussian_kernel_sum * data_loader.dataset.facofmaxgt_gp)
            # loss_self-supervision
            # loss_ssv = F.mse_loss(gp_map_res, view_gp_map_res.to(gp_map_res.device))
            loss_svp = F.mse_loss(view_gp_out, detect_gt.repeat(self.num_cam, 1, 1, 1).to(view_gp_out.device))
            if self.fix_svp:
                loss = loss_3D
            else:
                loss = loss_3D + self.weight_svp * loss_svp / self.num_cam
            losses += loss.item()
            Mae_3D_batches += mae_3D

            # Quick test
            # if batch_idx==2:
            #     break
        t1 = time.time()
        t_epoch = t1 - t0

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
            print(f'moda: {moda:.1f}%, modp: {modp:.1f}%, precision: {precision:.1f}%, recall: {recall:.1f}% ')

        Mae_3D_batches /= len(data_loader)
        losses /= len(data_loader)
        print(f'Time:{t_epoch:.3f}, Loss:{losses:.6f}, Mae_3D_batches:{Mae_3D_batches:.3f}')
        return losses, moda

    def middle_visualize_comparation(self, x, y, res_dir, title='None'):
        fig, axs = plt.subplots(nrows=1, ncols=2)
        fig.suptitle(title)
        axs[0].imshow(x)
        im = axs[1].imshow(y)
        fig.colorbar(im, ax=axs, fraction=0.05)
        plt.savefig(os.path.join(res_dir, title))
        plt.close(fig)

    def middle_visualize(self, x, res_dir, title='None'):
        pixel_max = x.max()
        pixel_min = x.min()
        norm = matplotlib.colors.Normalize(vmin=pixel_min, vmax=pixel_max)
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
        fig.suptitle(title)
        for i in range(self.num_cam):
            mappable_i = matplotlib.cm.ScalarMappable(norm=norm)
            im = axs[i].imshow(x[i].detach().cpu().squeeze(), norm=norm)
            # axs[i].set_xticks([])
            # axs[i].set_yticks([])
            axs[i].set_title(f'View_{i + 1}')
        fig.colorbar(mappable_i, ax=axs)
        plt.savefig(os.path.join(res_dir, title))
        # plt.show()
        plt.close(fig)

    def pretrained_model_loading_test(self, data_loader, epoch_resdir, visualize=True):
        for batch_idx, (data, detect_gt, frame) in enumerate(data_loader):
            gp_map_res, view_gp_map_res, w_mask = self.model(data)

            if visualize and batch_idx % 100 == 0:
                self.middle_visualize_comparation(detect_gt.cpu().squeeze(),
                                                  gp_map_res.detach().cpu().squeeze(),
                                                  epoch_resdir,
                                                  title=f'Loss_3D_density_map_frame{frame}')
                self.middle_visualize_comparation(view_gp_map_res.detach().cpu().squeeze(),
                                                  gp_map_res.detach().cpu().squeeze(),
                                                  epoch_resdir,
                                                  title=f'Loss_SSV_dmapfrom_feature_vs_dmapfrom_viewdmap_frame{frame}')
                self.middle_visualize(w_mask.detach().cpu(), epoch_resdir,
                                      title=f'Weight_Mask_frame{frame}')
