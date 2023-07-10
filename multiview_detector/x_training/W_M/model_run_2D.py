import os
import shutil

os.environ['OMP_NUM_THREADS'] = '1'
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
from multiview_detector.datasets import W_M

from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.datasets.W_M.frameDataset_head import frameDataset
import multiview_detector.datasets.W_M.MultiviewX
import multiview_detector.datasets.W_M.Wildtrack
import multiview_detector.datasets.W_M.frameDataset
from multiview_detector.models.W_M.persp_trans_detector_2D import PerspTransDetector_2D
from multiview_detector.trainer.W_M.trainer_2D import PerspectiveTrainer_2D
from torch.utils.tensorboard import SummaryWriter


def model_run(args):
    # dataset normlization
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize, ])
    # train_trans = T.Compose([T.ToTensor(), normalize])
    if 'wildtrack' in args.dataset:
        data_path = os.path.expanduser('~/Data/Wildtrack')
        base = W_M.Wildtrack.Wildtrack(data_path)
    elif 'multiviewx' in args.dataset:
        data_path = os.path.expanduser('~/Data/MultiviewX')
        base = W_M.MultiviewX.MultiviewX(data_path)
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
    train_set = frameDataset(base, train=True, transform=train_trans, grid_reduce=4)
    test_set = frameDataset(base, train=False, transform=train_trans, grid_reduce=4)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    # logging
    logdir = f'logs/{args.dataset}_frame/{args.arch}/{args.variant}/' + datetime.datetime.today().strftime(
        '%Y-%m-%d_%H-%M-%S') + f'momentum{args.momentum}_weight_decay{args.weight_decay}_lr{args.lr}_' \
                               f'lrs{args.lr_scheduler}_epo{args.epochs}'\
        if not args.resume else f'logs/{args.dataset}_frame/_{args.arch}/{args.variant}/{args.resume}'

    if args.resume is None:
        os.makedirs(logdir, exist_ok=True)
        shutil.copytree('/home/yunfei/Study/MVD_VCW/multiview_detector/datasets/W_M', logdir + '/scripts/datasets')
        shutil.copy('/home/yunfei/Study/MVD_VCW/multiview_detector/models/W_M/persp_trans_detector_2D.py',
                    logdir + '/scripts/persp_trans_detector_2D.py')
        shutil.copy('/home/yunfei/Study/MVD_VCW/multiview_detector/trainer/W_M/trainer_2D.py', logdir + '/scripts/trainer_2D.py')
        shutil.copy('/home/yunfei/Study/MVD_VCW/multiview_detector/x_training/W_M/model_run_2D.py',
                    logdir + '/scripts/model_run_2D.py')
        shutil.copy('/home/yunfei/Study/MVD_VCW/main.py', logdir + '/scripts/main.py')
    sys.stdout = Logger(os.path.join(logdir, 'log.txt.txt'), )
    print('Settings:')
    print(vars(args))
    writer = SummaryWriter(logdir + '/tensorboard/')  # 建立一个保存数据用的东西，save是输出的文件

    # model
    if args.variant == '2D':
        model = PerspTransDetector_2D(train_set, args.arch)
        trainer_2D = PerspectiveTrainer_2D(model, logdir, denormalize)

    else:
        raise Exception('no support for this variant')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.lr_scheduler == 'lambda':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lambda epoch: 1 / (
                                                              1 + args.lr_decay * epoch) ** epoch)
    elif args.lr_scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,  max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                        epochs=args.epochs)
    else:
        raise Exception('Must choose from [lambda, onecycle]')

    if args.resume is None:
        # print('Pretrained model get starting loading.......')
        # pretrained_model_dir = '/home/yunfei/Study/Baseline_dataset_multiviewX_WildTrack/logs/wildtrack_' \
        #                        'frame/2D/2023-03-03_23-40-35/MultiviewDetector.pth'
        # pretrained_model = torch.load(pretrained_model_dir, map_location='cuda:0')
        # model_dict = model.state_dict()
        # # pretrained_dict = pretrained_model['net']
        # pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
        # epoch_dir = os.path.join(logdir, f'alljpgs/First_test')
        # os.makedirs(epoch_dir)
        trainer_2D.test(test_loader, None, True)
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')
            train_loss = trainer_2D.train(epoch, train_loader, optimizer, args.log_interval, scheduler)
            writer.add_scalar(tag="train_loss", scalar_value=train_loss, global_step=epoch)

            if epoch % 5 == 0:
                epoch_dir = os.path.join(logdir, f'alljpgs/epoch-{epoch}_results')
                os.makedirs(epoch_dir)
            else:
                epoch_dir=None
            test_loss = trainer_2D.test(test_loader, epoch_dir, True)
            writer.add_scalar(tag="test_loss", scalar_value=test_loss, global_step=epoch)
            # save
            checkpoint = {
                'epoch': epoch,
                'net': model.state_dict(),
                'optim': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(logdir, 'MultiviewDetector.pth'))
    else:
        resume_dir = f'logs/{args.dataset}_frame/{args.arch}/{args.variant}/' + args.resume
        resume_fname = resume_dir + '/MultiviewDetector.pth'
        resume_model=torch.load(resume_fname)
        start_epoch=resume_model['epoch']
        model.load_state_dict(resume_model['net'])
        print(f'补充训练：取小学习率lr={args.lr}')
        for epoch in tqdm.tqdm(range(start_epoch+1, args.epochs + 1)):
            print('Training...')
            train_loss = trainer_2D.train(epoch, train_loader, optimizer, args.log_interval, scheduler)
            writer.add_scalar(tag="train_loss", scalar_value=train_loss, global_step=epoch)

            if epoch % 5 == 0:
                epoch_dir = os.path.join(logdir, f'alljpgs/epoch-{epoch}_results')
                os.makedirs(epoch_dir)
            else:
                epoch_dir=None
            test_loss = trainer_2D.test(test_loader, epoch_dir, True)
            writer.add_scalar(tag="test_loss", scalar_value=test_loss, global_step=epoch)
            # save
            checkpoint = {
                'epoch': epoch,
                'net': model.state_dict(),
                'optim': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(logdir, 'MultiviewDetector.pth'))
