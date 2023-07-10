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
        self.weight_ssv = kwargs['weight_ssv']
        self.fix_2D = kwargs['fix_2D']
        self.fix_svp = kwargs['fix_svp']
        self.fix_2D = kwargs['fix_2D']
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
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        Mae_3D_batches = 0
        # views_mask is used to cover gp_patch_views_dmap to produce loss_3D
        for batch_idx, (imgs, img_gt, masked_view_gp_gt, gp_gt, frame) in enumerate(data_loader):
            img_gt = img_gt.permute(1, 0, 2, 3)
            masked_view_gp_gt = masked_view_gp_gt.permute(1, 0, 2, 3)

            optimizer.zero_grad()
            # img_res, view_gp_output = self.model(imgs)
            img_res, view_gp_output = self.model(imgs)

            t_f = time.time()
            t_forward += t_f - t_b
            # loss 2D
            loss_2D = F.mse_loss(img_res, img_gt.to(img_res.device))
            # loss_SVP
            loss_svp = F.mse_loss(view_gp_output, masked_view_gp_gt.to(view_gp_output.device))

            loss = loss_2D + loss_svp * self.weight_svp
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
                      f'loss_2D:{loss_2D:.6f}, loss_svp:{loss_svp:.6f}, lr:{current_lr:.8f} '
                      f'Time: {t_epoch:.1f} (f{t_forward / batch_idx:.3f}+b{t_backward / batch_idx:.3f}), '
                      f'maxima:{view_gp_output.max():.3f}')
            if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                cyclic_scheduler.step()
        if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.LambdaLR):
            cyclic_scheduler.step()  # For Lambda

        t1 = time.time()
        t_epoch = t1 - t0
        losses /= len(data_loader)
        Mae_3D_batches /= len(data_loader)
        print(f'Train Epoch: {epoch}, Batch:{len(data_loader)}, Loss: {losses:.6f}, '
              f'Time: {t_epoch:.3f} ')
        return losses

    def test(self, data_loader, visualize=False, epoch_resdir=None, writer=None, first_test=False):
        self.model.eval()
        losses = 0
        t0 = time.time()
        MAE = 0
        for batch_idx, (imgs, img_gt, masked_view_gp_gt, gp_gt, frame) in enumerate(data_loader):
            frame = int(frame)
            img_gt = img_gt.permute(1, 0, 2, 3)
            masked_view_gp_gt = masked_view_gp_gt.permute(1, 0, 2, 3)
            with torch.no_grad():
                img_res, view_gp_output = self.model(imgs)
            # loss 2D
            loss_2D = F.mse_loss(img_res, img_gt.to(img_res.device))
            # loss_SVP
            loss_svp = F.mse_loss(view_gp_output, masked_view_gp_gt.to(view_gp_output.device))
            mae = abs(view_gp_output.sum() - masked_view_gp_gt.sum())
            MAE += mae
            loss = loss_2D + loss_svp * self.weight_svp
            losses += loss.item()

            # visualization
            if visualize and batch_idx % 100 == 0:
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
        t1 = time.time()
        MAE = MAE / len(data_loader) / self.num_cam
        print(f'MAE={MAE:.3f}')
        print(f'Test time:{t1 - t0:.1f}, losses:{losses:.6f}')
        return losses
    # def middle_visualize_comparation(self, x, y, res_dir, title='None'):
    #     fig, axs = plt.subplots(nrows=1, ncols=2)
    #     fig.suptitle(title)
    #     axs[0].imshow(x)
    #     im = axs[1].imshow(y)
    #     fig.colorbar(im, ax=axs, fraction=0.05)
    #     plt.savefig(os.path.join(res_dir, title))
    #     plt.close(fig)
    #
    # def middle_visualize(self, x, res_dir, title='None'):
    #     pixel_max = x.max()
    #     pixel_min = x.min()
    #     norm = matplotlib.colors.Normalize(vmin=pixel_min, vmax=pixel_max)
    #     fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
    #     fig.suptitle(title)
    #     for i in range(self.num_cam):
    #         mappable_i = matplotlib.cm.ScalarMappable(norm=norm)
    #         im = axs[i].imshow(x[i].detach().cpu().squeeze(), norm=norm)
    #         # axs[i].set_xticks([])
    #         # axs[i].set_yticks([])
    #         axs[i].set_title(f'View_{i + 1}')
    #     fig.colorbar(mappable_i, ax=axs)
    #     plt.savefig(os.path.join(res_dir, title))
    #     # plt.show()
    #     plt.close(fig)
    #
    # def pretrained_model_loading_test(self, data_loader, epoch_resdir, visualize=True):
    #     for batch_idx, (data, detect_gt, frame) in enumerate(data_loader):
    #         gp_map_res, view_gp_map_res, w_mask = self.model(data)
    #
    #         if visualize and batch_idx % 100 == 0:
    #             self.middle_visualize_comparation(detect_gt.cpu().squeeze(),
    #                                               gp_map_res.detach().cpu().squeeze(),
    #                                               epoch_resdir,
    #                                               title=f'Loss_3D_density_map_frame{frame}')
    #             self.middle_visualize_comparation(view_gp_map_res.detach().cpu().squeeze(),
    #                                               gp_map_res.detach().cpu().squeeze(),
    #                                               epoch_resdir,
    #                                               title=f'Loss_SSV_dmapfrom_feature_vs_dmapfrom_viewdmap_frame{frame}')
    #             self.middle_visualize(w_mask.detach().cpu(), epoch_resdir,
    #                                   title=f'Weight_Mask_frame{frame}')
