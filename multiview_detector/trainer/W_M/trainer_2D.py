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
from multiview_detector.utils.person_help import vis


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer_2D(BaseTrainer):
    def __init__(self, model, logdir, denormalize):
        super(BaseTrainer, self).__init__()
        self.model = model
        # self.criterion = criterion
        self.logdir = logdir
        self.denormalize = denormalize
        self.current_dir = self.logdir + '/alljpgs/current_epoch'
        os.makedirs(self.current_dir, exist_ok=True)

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0

        for batch_idx, (data, map_gt, img_gt, _) in enumerate(data_loader):
            optimizer.zero_grad()
            img_res = self.model(data)
            t_f = time.time()
            t_forward += t_f - t_b
            # gaussian_img_gt = target_transform(img_res, img_gt[0], data_loader.dataset.img_kernel)
            # loss = F.mse_loss(img_res, gaussian_img_gt.to(img_res.device)) / data_loader.dataset.num_cam
            loss=F.mse_loss(img_res,img_gt[0].to(img_res.device))/data_loader.dataset.num_cam
            loss.backward()
            optimizer.step()
            losses += loss.item()

            t_b = time.time()
            t_backward += t_b - t_f

            if (batch_idx + 1) % log_interval == 0:
                t1 = time.time()
                t_epoch = t1 - t0
                current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                print(
                    f'Train Epoch: {epoch}, Batch:{(batch_idx + 1)}, Loss: {losses / (batch_idx + 1):.6f} '
                    f'lr:{current_lr:.6f} Time: {t_epoch:.1f} (f{t_forward / batch_idx:.3f}+b{t_backward / batch_idx:.3f})')
            if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                cyclic_scheduler.step()
        if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.LambdaLR):
            cyclic_scheduler.step()  # For Lambda

        t1 = time.time()
        t_epoch = t1 - t0
        losses /= len(data_loader)
        print(f'Train Epoch: {epoch}, Batch:{len(data_loader)}, Loss: {losses:.6f}, Time: {t_epoch:.3f}')
        return losses

    def test(self, data_loader, epoch_resdir, visualize=True):
        print('Testing...')
        self.model.eval()
        losses = 0
        t0 = time.time()

        for batch_idx, (data, map_gt, img_gt, frame) in enumerate(data_loader):
            with torch.no_grad():
                img_res = self.model(data)

            # gaussian_img_gt = target_transform(img_res, img_gt[0], data_loader.dataset.img_kernel)
            # loss = F.mse_loss(img_res, gaussian_img_gt.to(img_res.device)) / data_loader.dataset.num_cam
            loss=F.mse_loss(img_res,img_gt[0].to(img_res.device))/data_loader.dataset.num_cam
            losses += loss.item()
            # visualizing the heatmap for per-view estimation
            if visualize and batch_idx % 20 == 0:
                if epoch_resdir is None:
                    epoch_resdir = self.current_dir
                heatmap0_head = img_res[0, 0].detach().cpu().numpy().squeeze()
                # heatmap0_foot = img_res[0, 1].detach().cpu().numpy().squeeze()
                img0 = self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
                img0 = Image.fromarray((img0 * 255).astype('uint8'))
                head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
                head_cam_result.save(os.path.join(epoch_resdir, 'cam1_head.jpg'))
                # foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
                # foot_cam_result.save(os.path.join(epoch_resdir, 'cam1_foot.jpg'))

        t1 = time.time()
        losses /= len(data_loader)
        print(f'Test time:{t1 - t0:.1f}, losses:{losses:.6f}')
        return losses
# class BBOXTrainer(BaseTrainer):
#     def __init__(self, model, criterion, cls_thres):
#         super(BaseTrainer, self).__init__()
#         self.model = model
#         self.criterion = criterion
#         self.cls_thres = cls_thres
#
#     def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
#         self.model.train()
#         losses = 0
#         correct = 0
#         miss = 0
#         t0 = time.time()
#         for batch_idx, (data, target, _) in enumerate(data_loader):
#             data, target = data.cuda(), target.cuda()
#             optimizer.zero_grad()
#             output = self.model(data)
#             pred = torch.argmax(output, 1)
#             correct += pred.eq(target).sum().item()
#             miss += target.numel() - pred.eq(target).sum().item()
#             loss = self.criterion(output, target)
#             loss.backward()
#             optimizer.step()
#             losses += loss.item()
#             if cyclic_scheduler is not None:
#                 if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
#                     cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
#                 elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
#                     cyclic_scheduler.step()
#             if (batch_idx + 1) % log_interval == 0:
#                 # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
#                 t1 = time.time()
#                 t_epoch = t1 - t0
#                 print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
#                     epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))
#
#         t1 = time.time()
#         t_epoch = t1 - t0
#         print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
#             epoch, len(data_loader), losses / len(data_loader), 100. * correct / (correct + miss), t_epoch))
#
#         return losses / len(data_loader), correct / (correct + miss)
#
#     def test(self, test_loader, log_interval=100, res_fpath=None):
#         self.model.eval()
#         losses = 0
#         correct = 0
#         miss = 0
#         all_res_list = []
#         t0 = time.time()
#         for batch_idx, (data, target, (frame, pid, grid_x, grid_y)) in enumerate(test_loader):
#             data, target = data.cuda(), target.cuda()
#             with torch.no_grad():
#                 output = self.model(data)
#                 output = F.softmax(output, dim=1)
#             pred = torch.argmax(output, 1)
#             correct += pred.eq(target).sum().item()
#             miss += target.numel() - pred.eq(target).sum().item()
#             loss = self.criterion(output, target)
#             losses += loss.item()
#             if res_fpath is not None:
#                 indices = output[:, 1] > self.cls_thres
#                 all_res_list.append(torch.stack([frame[indices].float(), grid_x[indices].float(),
#                                                  grid_y[indices].float(), output[indices, 1].cpu()], dim=1))
#             if (batch_idx + 1) % log_interval == 0:
#                 # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
#                 t1 = time.time()
#                 t_epoch = t1 - t0
#                 print('Test Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
#                     (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))
#
#         t1 = time.time()
#         t_epoch = t1 - t0
#         print('Test, Batch:{}, Loss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
#             len(test_loader), losses / (len(test_loader) + 1), 100. * correct / (correct + miss), t_epoch))
#
#         if res_fpath is not None:
#             all_res_list = torch.cat(all_res_list, dim=0)
#             np.savetxt(os.path.dirname(res_fpath) + '/all_res.txt', all_res_list.numpy(), '%.8f')
#             res_list = []
#             for frame in np.unique(all_res_list[:, 0]):
#                 res = all_res_list[all_res_list[:, 0] == frame, :]
#                 positions, scores = res[:, 1:3], res[:, 3]
#                 ids, count = nms(positions, scores, )
#                 res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
#             res_list = torch.cat(res_list, dim=0).numpy()
#             np.savetxt(res_fpath, res_list, '%d')
#
#         return losses / len(test_loader), correct / (correct + miss)
