import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import datasets.W_M.MultiviewX
import datasets.W_M.Wildtrack
import datasets.W_M.frameDataset

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
from datasets import *

from utils.logger import Logger
from utils.draw_curve import draw_curve
from utils.image_utils import img_color_denormalize
from datasets.W_M.frameDataset import frameDataset
from models.W_M.persp_trans_detector_2D import PerspTransDetector_2D
from models.W_M.persp_trans_detector_2D_SVP import PerspTransDetector_2D_SVP
from trainer.W_M.trainer_2D import PerspectiveTrainer_2D
from trainer.W_M.trainer_2D_SVP import PerspectiveTrainer_2D_SVP
from torch.utils.tensorboard import SummaryWriter

device = ['cuda:3', 'cuda:3']


def model_run(args):
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize, ])
    # train_trans = T.Compose([T.ToTensor(), normalize])
    if 'wildtrack' in args.dataset:
        data_path = os.path.expanduser('~/Data/Wildtrack')
        base = datasets.W_M.Wildtrack.Wildtrack(data_path)
    elif 'multiviewx' in args.dataset:
        data_path = os.path.expanduser('~/Data/MultiviewX')
        base = datasets.W_M.MultiviewX.MultiviewX(data_path)
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
    train_set = frameDataset(base, train=True, transform=train_trans, grid_reduce=4)
    test_set = frameDataset(base, train=False, transform=train_trans, grid_reduce=4)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    # logging
    logdir = f'logs/{args.dataset}_frame/_{args.arch}/_{args.variant}/' + datetime.datetime.today().strftime(
        '%Y-%m-%d_%H-%M-%S') + f'_fix2D{args.fix_2D}w{args.weight_2D}_fixsvp{args.fix_svp}w{args.weight_svp}_' \
                               f'momentum{args.momentum}_weight_decay{args.weight_decay}_lr{args.lr}_lrs{args.lr_scheduler}_epo{args.epochs}_' \
                               f'ct{args.cls_thres}_nt{args.nms_thres}_dt{args.dist_thres}' \
        if not args.resume else f'logs/{args.dataset}_frame/{args.variant}/{args.resume}'
    if args.resume is None:
        os.makedirs(logdir, exist_ok=True)
        copy_tree('./multiview_detector', logdir + '/scripts/multiview_detector')
        for script in os.listdir('.'):
            if script.split('.')[-1] == 'py':
                dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)

    sys.stdout = Logger(os.path.join(logdir, 'log.txt.txt'), )
    print('Settings:')
    print(vars(args))
    writer = SummaryWriter(f'tensorboard/{logdir}')  # 建立一个保存数据用的东西，save是输出的文件

    # model
    if args.variant == '2D':
        model = PerspTransDetector_2D(train_set, args.arch)
        trainer_2D = PerspectiveTrainer_2D(model, logdir, denormalize)
    elif args.variant == '2D_SVP':
        model = PerspTransDetector_2D_SVP(train_set, args.arch)
        trainer_2D_SVP = PerspectiveTrainer_2D_SVP(model, logdir, denormalize, fix_svp=args.fix_svp, fix_2D=args.fix_2D,
                                                   weight_svp=args.weight_svp, weight_2D=args.weight_2D)

    else:
        raise Exception('no support for this variant')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.lr_scheduler == 'lambda':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lambda epoch: 1 / (
                                                              1 + args.lr_decay * epoch) ** epoch)
    elif args.lr_scheduler == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                        epochs=args.epochs)
    else:
        raise Exception('Must choose from [lambda, onecycle]')

    if args.resume is None:
        print('Pretrained model get starting loading.......')
        pretrained_model_dir = '/home/yunfei/Study/Baseline_dataset_multiviewX_WildTrack/logs/wildtrack_' \
                               'frame/2D/2023-03-03_23-40-35/MultiviewDetector.pth'
        pretrained_model = torch.load(pretrained_model_dir, map_location=device[0])
        model_dict = model.state_dict()
        # pretrained_dict = pretrained_model['net']
        pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        epoch_dir = os.path.join(logdir, f'alljpgs/First_test')
        os.makedirs(epoch_dir)
        trainer_2D_SVP.test(test_loader, epoch_dir, True)
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')
            train_loss = trainer_2D_SVP.train(epoch, train_loader, optimizer, args.log_interval, scheduler)
            writer.add_scalar(tag="train_loss", scalar_value=train_loss, global_step=epoch)

            if epoch % 5 == 0 or epoch <= 2:
                epoch_dir = os.path.join(logdir, f'alljpgs/epoch-{epoch}_results')
                os.makedirs(epoch_dir)
                test_loss = trainer_2D_SVP.test(test_loader, epoch_dir, True)
                writer.add_scalar(tag="test_loss", scalar_value=test_loss, global_step=epoch)
            # save
            checkpoint = {
                'epoch': epoch,
                'net': model.state_dict(),
                'optim': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(logdir, 'MultiviewDetector.pth'))
    else:
        resume_dir = f'logs/{args.dataset}_frame/{args.variant}/' + args.resume
        resume_fname = resume_dir + '/MultiviewDetector.pth'
        model.load_state_dict(torch.load(resume_fname))
        print(f'补充训练：取小学习率lr={args.lr}')
        for epoch in range(101, args.epochs):
            print('Training...')
            train_loss = trainer_2D_SVP.train(epoch, train_loader, optimizer, args.log_interval, scheduler)
            writer.add_scalar(tag="train_loss", scalar_value=train_loss, global_step=epoch)
            if epoch % 5 == 0 or epoch == args.epochs:
                epoch_dir = os.path.join(logdir, f'alljpgs/epoch-{epoch}_results')
                os.makedirs(epoch_dir)
                test_loss = trainer_2D_SVP.test(test_loader, epoch_dir, True)
                writer.add_scalar(tag="test_loss", scalar_value=test_loss, global_step=epoch)
            # save
            checkpoint = {
                'epoch': epoch,
                'net': model.state_dict(),
                'optim': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(logdir, 'MultiviewDetector.pth'))


