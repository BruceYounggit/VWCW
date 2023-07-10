import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T

from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.datasets.CityStreet.Citystreet import Citystreet
from multiview_detector.datasets.CityStreet.framedataset_depthmap import frameDataset_depth_map_full_size
from multiview_detector.models.CityStreet.Perspective_depthmap_2D_SVP_WS import DPerspTransDetector
from multiview_detector.trainer.CityStreet.trainer_depthmap_2D_SVP_WS import PerspectiveTrainer
from torch.utils.tensorboard import SummaryWriter


def model_run(args):
    # seed
    if sys.gettrace() is None:
        print("Not in debug mode")
    else:
        print("In debug mode")
    debug = sys.gettrace()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([1520 // args.img_reduce, 2704 // args.img_reduce]), T.ToTensor(), normalize])
    if 'citystreet' in args.dataset:
        data_path = os.path.expanduser('~/Data/CityStreet')
        base = Citystreet(data_path)
    else:
        raise Exception('must choose from [wildtrack, multiviewx, citystreet]')

    # model & DATASET
    train_set = frameDataset_depth_map_full_size(base, train=True, transform=train_trans, force_download=True,
                                                 map_sigma=args.map_sigma,
                                                 world_reduce=args.world_reduce, img_deduce=args.img_reduce,
                                                 facofmaxgt=args.facofmaxgt, facofmaxgt_gp=args.facofmaxgt_gp)

    test_set = frameDataset_depth_map_full_size(base, train=False, transform=train_trans, force_download=True,
                                                map_sigma=args.map_sigma,
                                                world_reduce=args.world_reduce, img_deduce=args.img_reduce,
                                                facofmaxgt=args.facofmaxgt, facofmaxgt_gp=args.facofmaxgt_gp)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)
    # Note that the coefficient "fix_weight", which is used for fix the MLP of learning weight from depth map.
    model = DPerspTransDetector(train_set, arch=args.arch, fix_2D=args.fix_2D,
                                fix_svp=args.fix_svp, fix_weight=args.fix_weight, person_heights=args.person_heights)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr[0], momentum=args.momentum, weight_decay=args.weight_decay)
    if 'resnet' in args.arch:
        optimizer1 = optim.SGD([{'params': model.base_pt1.parameters(), 'lr': args.lr},
                                {'params': model.base_pt2.parameters(), 'lr': args.lr},
                                {'params': model.img_classifier.parameters(), 'lr': args.lr},
                                {'params': model.confidence_attention.parameters(), 'lr': args.lr},
                                {'params': model.weight_calculation.parameters(), 'lr': args.lr},
                                {'params': model.fusion_net.parameters(), 'lr': args.lr}
                                ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer1 = optim.SGD([{'params': model.base_pt.parameters(), 'lr': args.lr},
                                {'params': model.img_classifier.parameters(), 'lr': args.lr},
                                {'params': model.confidence_attention.parameters(), 'lr': args.lr},
                                {'params': model.weight_calculation.parameters(), 'lr': args.lr},
                                {'params': model.fusion_net.parameters(), 'lr': args.lr}
                                ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler1 = optim.lr_scheduler.OneCycleLR(optimizer1, max_lr=args.lr,
                                               steps_per_epoch=len(train_loader),
                                               epochs=args.epochs)
    optimizer2 = optim.SGD([{'params': model.view_gp_decoder.parameters(), 'lr': args.lr * args.lrfac},
                            {'params': model.GP_Decoder.parameters(), 'lr': args.lr * args.lrfac},
                            ], momentum=args.momentum, weight_decay=args.weight_decay, lr=args.lr * args.lrfac)
    # scheduler2 = optim.lr_scheduler.LambdaLR(optimizer2,
    #                                          lr_lambda=lambda epoch: 1 / (1 + args.lr_decay * epoch) ** epoch)
    scheduler2 = optim.lr_scheduler.OneCycleLR(optimizer2, max_lr=args.lr,
                                               steps_per_epoch=len(train_loader),
                                               epochs=args.epochs)
    optimizer = {'optimizer1': optimizer1,
                 'optimizer2': optimizer2}
    scheduler = {'scheduler1': scheduler1,
                 'scheduler2': scheduler2}
    #
    # if args.lr_scheduler == 'lambda':
    #     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    #                                                   lr_lambda=lambda epoch: 1 / (
    #                                                           1 + args.lr_decay * epoch) ** epoch)
    # elif args.lr_scheduler == 'onecycle':
    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
    #                                                     steps_per_epoch=len(train_loader),
    #                                                     epochs=args.epochs)
    # else:
    #     raise Exception('Must choose from [lambda, onecycle]')

    # -----------------------------------------------------------------------------------

    # logging
    logdir = f'logs/{args.dataset}_frame/{args.arch}/{args.variant}/' + datetime.datetime.today().strftime(
        '%Y-%m-%d_%H-%M-%S') + f'_fix2D{args.fix_2D}w{args.weight_2D}_fixsvp{args.fix_svp}w{args.weight_svp}_' \
                               f'momentum{args.momentum}_weight_decay{args.weight_decay}_lr{args.lr}_lrs{args.lr_scheduler}_epo{args.epochs}_' \
                               f'ct{args.cls_thres}_nt{args.nms_thres}_dt{args.dist_thres}' \
        if not args.resume else f'logs/{args.dataset}_frame/{args.arch}/{args.variant}/{args.resume}'

    if args.resume is None:
        os.makedirs(logdir, exist_ok=True)
        shutil.copytree('/home/yunfei/Study/MVD_VCW/multiview_detector/datasets/CityStreet',
                        logdir + '/scripts/datasets')
        shutil.copytree('/home/yunfei/Study/MVD_VCW/multiview_detector/models/CityStreet',
                        logdir + '/scripts/models')
        shutil.copy('/home/yunfei/Study/MVD_VCW/multiview_detector/models/CityStreet/Perspective_depthmap_2D_SVP_WS.py',
                    logdir + '/scripts/Perspective_depthmap_2D_SVP_WS.py')
        shutil.copy('/home/yunfei/Study/MVD_VCW/multiview_detector/trainer/CityStreet/trainer_depthmap_2D_SVP_WS.py',
                    logdir + '/scripts/trainer_2D_SVP_WS.py')
        shutil.copy('/home/yunfei/Study/MVD_VCW/multiview_detector/x_training/CityStreet/model_run_2D_SVP_WS.py',
                    logdir + '/scripts/model_run_2D_SVP_WS.py')
        shutil.copy('/home/yunfei/Study/MVD_VCW/main.py', logdir + '/scripts/main.py')

    # tensorboard wtiter
    sys.stdout = Logger(os.path.join(logdir, 'log.txt.txt'), )
    print('Settings:')
    print(vars(args))
    writer = SummaryWriter(logdir + '/tensorboard')

    # Trainer
    trainer = PerspectiveTrainer(model, logdir, denormalize, args.cls_thres * train_set.facofmaxgt_gp, args.nms_thres,
                                 args.dist_thres, weight_2D=args.weight_2D, weight_svp=args.weight_svp,
                                 fix_2D=args.fix_2D, fix_svp=args.fix_svp, fix_weight=args.fix_weight)
    # learn

    if args.resume is None:
        print('Pretrained model get starting loading.......')
        if args.arch == 'resnet18':
            # pretrained_model_dir = '/home/yunfei/Study/MVD_VCW/logs/citystreet_frame/resnet18/2D_SVP/2023-04-23_21-47-16_fix2D1.0w1_fixsvp0w1_momentum0.9_' \
            #                        'weight_decay0.0001_lr0.01_lrsonecycle_epo100_ct0.4_nt20_dt20/MultiviewDetector.pth'
            # pretrained_model_dir = '/home/yunfei/Study/MVD_VCW/logs/citystreet_frame/resnet18/2D_SVP_WS/2023-04-24_15-49-05_fix2D1' \
            #                        '.0w1_fixsvp1.0w1_momentum0.9_weight_decay0.0001_lr0.01_lrsonecycle_epo100' \
            #                        '_ct0.4_nt10_dt20/MultiviewDetector.pth'
            pretrained_model_dir = '/home/yunfei/Study/MVD_VCW/logs/citystreet_frame/resnet18/2D_SVP_WS/2023-04-27_10-19-13_fix2D1.0w1_fixsv' \
                                   'p0.0w1_momentum0.9_weight_decay0.0001_lr0.0001_lrslambda_epo200_ct0.4_nt10_dt20/MultiviewDetector.pth'
        else:
            # pretrained_model_dir = '/home/yunfei/Study/MVD_VCW/logs/citystreet_frame/vgg16/2D_SVP/2023-04-24_19-12-45_fix2D1.0w1_
            # fixsvp0.0w1_momentum0.9_weight_decay0.0001_lr0.01_lrsonecycle_epo100_ct0.4_nt10_dt20/MultiviewDetector_100.pth'
            pretrained_model_dir = '/home/yunfei/Study/MVD_VCW/logs/citystreet_frame/vgg16/2D_SVP_WS/2023-04-27_10-18-04_fix2D1.0w1_fixsvp0' \
                                   '.0w1_momentum0.9_weight_decay0.0001_lr0.0001_lrslambda_epo200_ct0.4_nt10_dt20/MultiviewDetector.pth'

        pretrained_model = torch.load(pretrained_model_dir, map_location='cuda:0')
        model_dict = model.state_dict()
        pretrained_dict = pretrained_model['net']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('Start training...')
        epoch_dir = os.path.join(logdir, f'alljpgs/First_test')
        os.makedirs(epoch_dir, exist_ok=True)
        trainer.test(test_loader, train_set.gt_fpath, True, epoch_dir, first_test=True)
        print('\n')
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            train_loss = trainer.train(train_loader, epoch, optimizer, args.log_interval, scheduler, writer)
            writer.add_scalar(tag="Ttain_loss", scalar_value=train_loss, global_step=epoch)
            # save
            checkpoint = {
                'epoch': epoch,
                'net': model.state_dict(),
                'optim1': optimizer['optimizer1'].state_dict(),
                'optim2': optimizer['optimizer2'].state_dict()
            }
            # if epoch % 50 == 0:
            #     torch.save(checkpoint, os.path.join(logdir, f'MultiviewDetector_{epoch}.pth'))
            # else:
            torch.save(checkpoint, os.path.join(logdir, f'MultiviewDetector.pth'))
            print('Testing...')
            if epoch % 5 == 0:
                epoch_resdir = os.path.join(logdir, f'alljpgs/epoch-{epoch}_results')
                os.makedirs(epoch_resdir, exist_ok=True)
                test_loss = trainer.test(test_loader, train_set.gt_fpath, True, epoch_resdir, first_test=False)
                writer.add_scalar(tag="Test_loss", scalar_value=test_loss, global_step=epoch)
    else:  # 继续训练
        resume_root = '/home/yunfei/Study/MVD_VCW'
        resume_dir = os.path.join(resume_root, f'logs/{args.dataset}_frame/{args.arch}/{args.variant}/{args.resume}')
        resume_fname = resume_dir + f'/MultiviewDetector.pth'
        # # 加载预训练模型
        checkpoint = torch.load(resume_fname)
        model.load_state_dict(checkpoint['net'])
        # optimizer.load_state_dict(checkpoint['optim'])
        start_epoch = checkpoint['epoch'] + 1
        # scheduler.load_state_dict(checkpoint['scheduler'])
        print('Load data successfully...')
        for epoch in tqdm.tqdm(range(start_epoch, args.epochs + 1)):
            train_loss = trainer.train(train_loader, epoch, optimizer, args.log_interval, scheduler, writer)
            writer.add_scalar(tag="Ttain_loss", scalar_value=train_loss, global_step=epoch)
            # save
            checkpoint = {
                'epoch': epoch,
                'net': model.state_dict(),
                'optim1': optimizer['optimizer1'].state_dict(),
                'optim2': optimizer['optimizer2'].state_dict()
            }
            # if epoch % 50 == 0:
            #     torch.save(checkpoint, os.path.join(logdir, f'MultiviewDetector_{epoch}.pth'))
            # else:
            torch.save(checkpoint, os.path.join(logdir, f'MultiviewDetector.pth'))
            print('Testing...')
            if epoch % 5 == 0:
                epoch_resdir = os.path.join(logdir, f'alljpgs/epoch-{epoch}_results')
                os.makedirs(epoch_resdir, exist_ok=True)
                test_loss = trainer.test(test_loader, train_set.gt_fpath, True, epoch_resdir, first_test=True)
                writer.add_scalar(tag="Test_loss", scalar_value=test_loss, global_step=epoch)
    writer.close()  # 结束服务
