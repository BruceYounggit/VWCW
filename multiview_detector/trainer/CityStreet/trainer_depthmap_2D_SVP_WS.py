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
            img_res, view_gp_output, feat_fusion, pred_fusion, joint_fusion, w_mask = self.model(imgs)

            t_f = time.time()
            t_forward += t_f - t_b
            # loss_2D for RGB images
            loss_2D = F.mse_loss(img_res, img_gt.to(img_res.device))
            # loss_SVP for single-view prediction
            loss_svp = F.mse_loss(view_gp_output, masked_view_gp_gt.to(view_gp_output.device))
            # loss_fusion for the average value of weight summation of feature and weight summation of single-view prediction.
            # mean_out = (feat_fusion + pred_fusion) / 2  # average
            # joint_fusion = mean_out
            feat_fusion_loss = F.mse_loss(feat_fusion, gp_gt.to(feat_fusion.device))
            pred_fusion_loss = F.mse_loss(pred_fusion, gp_gt.to(pred_fusion.device))
            consistency_loss = F.mse_loss(pred_fusion, feat_fusion)
            loss_fusion = consistency_loss
            # loss_fusion = F.mse_loss(joint_fusion, gp_gt.to(joint_fusion.device))
            # loss_fusion = F.mse_loss(mean_out, gp_gt.to(mean_out.device))
            # loss_ssl for self-supervision about feat_fusion and pred_fusion
            # diff = feat_fusion_out - pred_fusion/ike(diff).to(diff.device))

            # loss = loss_2D * self.weight_2D + loss_svp * self.weight_svp + feat_fusion_loss + pred_fusion_loss
            # loss = (feat_fusion_loss + pred_fusion_loss) * 0.1 + consistency_loss
            loss = (feat_fusion_loss + pred_fusion_loss) * 0.1 + loss_fusion
            # loss = loss_fusion
            loss.backward()
            # optimizer.step()
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
                      f'2D:{loss_2D:.6f}, svp:{loss_svp:.6f}, feat_fusion:{feat_fusion_loss:.6f},'
                      f' pred_fusion:{pred_fusion_loss:.6f}, joint_fusion:{loss_fusion:.6f}, lr1:{current_lr[0]:.8f},lr2:{current_lr[1]:.8f}, '
                      f'Time: {t_epoch:.1f} (f{t_forward / batch_idx:.3f}+b{t_backward / batch_idx:.3f}), '
                      f'view_gp_max:{view_gp_output.max():.3f}, joint_fusion_maxima:{joint_fusion.max():.3f}')

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
        losses = pf_losses = 0
        t0 = time.time()
        all_res_lists = [[] for _ in range(3)]
        res_fpath = os.path.join(epoch_resdir, 'res.txt')
        for batch_idx, (imgs, img_gt, masked_view_gp_gt, gp_gt, frame) in enumerate(data_loader):
            frame = int(frame)
            # img_gt = img_gt.permute(1, 0, 2, 3)
            masked_view_gp_gt = masked_view_gp_gt.permute(1, 0, 2, 3)
            with torch.no_grad():
                img_res, view_gp_output, feat_fusion, pred_fusion, joint_fusion, w_mask = self.model(imgs)
                # img_res, view_gp_output, mean_out, w_mask = self.model(imgs)
                # mean_out = (feat_fusion + pred_fusion) / 2.0
                # joint_fusion = mean_out
                x_out = [pred_fusion, feat_fusion, joint_fusion]
                for i, out in enumerate(x_out):
                    map_grid_res = out.detach().cpu().squeeze()
                    v_s = map_grid_res[map_grid_res > self.cls_thres].unsqueeze(1)
                    grid_ij = (map_grid_res > self.cls_thres).nonzero()
                    all_res_lists[i].append(torch.cat([torch.ones_like(v_s) * frame, grid_ij.float() *
                                                       data_loader.dataset.world_reduce, v_s], dim=1))
            # loss_2D for RGB images
            # loss_2D = F.mse_loss(img_res, img_gt.to(img_res.device))
            # loss_SVP for single-view prediction
            # loss_svp = F.mse_loss(view_gp_output, masked_view_gp_gt.to(view_gp_output.device))
            # loss_fusion for the average value of weight summation of feature and weight summation of single-view prediction.
            feat_fusion_loss = F.mse_loss(feat_fusion, gp_gt.to(feat_fusion.device))
            pred_fusion_loss = F.mse_loss(pred_fusion, gp_gt.to(pred_fusion.device))
            consistency_loss = F.mse_loss(pred_fusion, feat_fusion)
            loss_fusion = consistency_loss
            # loss_fusion = F.mse_loss(joint_fusion, gp_gt.to(joint_fusion.device))
            # loss_fusion = F.mse_loss(mean_out, gp_gt.to(mean_out.device))
            # loss_ssl for self-supervision about feat_fusion and pred_fusion
            # diff = feat_fusion_out - pred_fusion/ike(diff).to(diff.device))

            # loss = loss_2D * self.weight_2D + loss_svp * self.weight_svp + feat_fusion_loss + pred_fusion_loss
            loss = (feat_fusion_loss + pred_fusion_loss) * 0.1 + loss_fusion
            # loss = loss_fusion
            losses += loss.item()
            # pf_losses +=

            # visualization
            if visualize and batch_idx % 100 == 0:
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
                        plt.imshow(w_mask[view].detach().squeeze().cpu().numpy())
                        plt.colorbar()
                        plt.savefig(os.path.join(epoch_resdir, f'weight_mask_frame{frame}_view{view}.jpg'))
                        plt.close()

                # Fusion vis
                fig = plt.figure()
                subplt0 = fig.add_subplot(221, title="fusion_target", xticks=[], yticks=[])
                subplt1 = fig.add_subplot(222, title='joint_fusion', xticks=[], yticks=[])
                subplt2 = fig.add_subplot(223, title='feat_fusion', xticks=[], yticks=[])
                subplt3 = fig.add_subplot(224, title='pred_fusion', xticks=[], yticks=[])
                subplt0.imshow(gp_gt.cpu().squeeze())
                subplt1.imshow(joint_fusion.cpu().squeeze().numpy())
                subplt2.imshow(feat_fusion.cpu().squeeze().numpy())
                subplt3.imshow(pred_fusion.cpu().squeeze().numpy())
                plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=-0.4, hspace=0.2)
                plt.savefig(os.path.join(epoch_resdir, f'Fusion_res_frame{frame}.jpg'), dpi=600)
                plt.close(fig)
                #
                # fig = plt.figure()
                # subplt0 = fig.add_subplot(121, title="fusion_target")
                # subplt1 = fig.add_subplot(122, title='feat_fusion')
                # subplt0.imshow(gp_gt.cpu().squeeze())
                # subplt1.imshow(feat_fusion.cpu().squeeze().numpy())
                # plt.savefig(os.path.join(epoch_resdir, f'feat_fusion_frame{frame}.jpg'))
                # plt.close(fig)
                #
                # fig = plt.figure()
                # subplt0 = fig.add_subplot(121, title="fusion_target")
                # subplt1 = fig.add_subplot(122, title='pred_fusion')
                # subplt0.imshow(gp_gt.cpu().squeeze())
                # subplt1.imshow(pred_fusion.cpu().squeeze().numpy())
                # plt.savefig(os.path.join(epoch_resdir, f'pred_fusion_frame{frame}.jpg'))
                # plt.close(fig)

        t1 = time.time()
        if res_fpath is not None:
            print('i==0: pred, i==1: feat, i==2: joint_fusion.')
            for i, all_res_list in enumerate(all_res_lists):
                all_res_list = torch.cat(all_res_list, dim=0)
                np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + f'/all_{i}_res.txt', all_res_list.numpy(),
                           '%.8f')
                for ntdt in [[10, 20]]:
                    res_list = []
                    nt, dt = ntdt[0], ntdt[1]
                    for frame in np.unique(all_res_list[:, 0]):
                        res = all_res_list[all_res_list[:, 0] == frame, :]
                        positions, scores = res[:, 1:3], res[:, 3]
                        ids, count = nms(positions, scores, nt, np.inf)
                        res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
                    res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
                    np.savetxt(res_fpath, res_list, '%d')
                    recall, precision, moda, modp = evaluateDetection_py(os.path.abspath(res_fpath),
                                                                         os.path.abspath(gt_fpath),
                                                                         dt,
                                                                         data_loader.dataset.base.__name__)
                    F1_score = 2 * precision * recall / (precision + recall + 1e-12)
                    print(f'When i={i}, nt={nt}, dt={dt}, moda: {moda:.1f}%, modp: {modp:.1f}%,'
                          f' precision: {precision:.1f}%, recall: {recall:.1f}%, F1_score:{F1_score:.1f}')

        losses = losses / len(data_loader)
        print(f'Test time:{t1 - t0:.1f}, losses:{losses:.6f}')
        return losses
