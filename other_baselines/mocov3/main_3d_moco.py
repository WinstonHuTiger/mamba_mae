#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

import argparse
import builtins
import math
import os
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pickle
import random
import time
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchio as tio
import warnings
from functools import partial
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('../../')

from utils import brats, brats_160, ct_176

import moco.builder
import moco.loader
import moco.optimizer


from moco.resent3d_base import generate_model
from moco.vit_3d import VisionTransformer3D
from models_vim import vim_3D_small_patch16_stride16_224_bimambav2_final_pool_mean_abs_pos_embed_div2, vim_mha_3D_small_patch16_stride16_final_pool_mean_abs_pos_embed_div2, vim2_3D_small_patch16_stride16_final_pool_mean_abs_pos_embed_div2

model_names = ['vit_3d', 'resnet_3d', 'vim_3d', "vim2_3d", "vim_mha_3d"]

parser = argparse.ArgumentParser(description='MoCo 3D Pre-Training')
# BEGIN:Changes specific to the medical dataset

# The dataset we are going to use
# print("This will only work with Brats and with the mode=whole")
parser.add_argument('--use_z_score', type=bool, default=True)
parser.add_argument('--split', type=str, default='idh', help="split for the large brats dataset")

parser.add_argument('--in_channels', type=int, default=1)
parser.add_argument('--num_classes', type=int, default=2)

parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument('--volume_size', default=96, type=int,
                    help='node rank for distributed training')
# END:Changes specific to the medical dataset

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet_3d',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=96, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.6, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--save_freq', default=20, type=int, help = "Frequency of saving the model")

# moco specific configs:
parser.add_argument('--moco-dim', default=256, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--moco-m', default=0.99, type=float,
                    help='moco momentum of updating momentum encoder (default: 0.99)')
parser.add_argument('--moco-m-cos', action='store_true',
                    help='gradually increase moco momentum to 1 with a '
                         'half-cycle cosine schedule')
parser.add_argument('--moco-t', default=1.0, type=float,
                    help='softmax temperature (default: 1.0)')

# vit specific configs:
parser.add_argument('--stop-grad-conv1', action='store_true',
                    help='stop-grad after first conv, or patch embedding')

# other upgrades
parser.add_argument('--optimizer', default='lars', type=str,
                    choices=['lars', 'adamw'],
                    help='optimizer used (default: lars)')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--crop-min', default=0.08, type=float,
                    help='minimum scale for random cropping (default: 0.08)')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        print("Using non-distributes single GPU training")
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
        
    model, optimizer = create_model_and_optim(args, ngpus_per_node)
    scaler = torch.cuda.amp.GradScaler()
    summary_writer = SummaryWriter() if args.rank == 0 else None
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']
            new_state_dict = dict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v  # remove `module.` prefix
                else:
                    new_state_dict[k] = v
            model.module.load_state_dict(new_state_dict)
            # model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # BEGIN: Changes specific to the medical dataset
    # build data

    # Since this is always brats, let us just hard-code it here.
    # Enables us to reuse the function
    args.name_dataset = 'brats-160'

    transforms = [
        tio.RandomAffine(),
        tio.RandomNoise(std=0.1),
        tio.RandomGamma(log_gamma=(-0.3, 0.3))
    ]
    
    transforms2 = [
        tio.RandomAffine(),
        # tio.RandomBiasField(coefficients=0.4),
        tio.RandomNoise(std=0.1),
        tio.RandomGamma(log_gamma=(-0.3, 0.3))
    ]
    
    train_transforms1 = tio.Compose(transforms)
    train_transforms2 = tio.Compose(transforms2)
    
    if args.name_dataset == 'brats':
        dataset_train = brats.build_dataset( mode='train',
                                            
                                            transforms=moco.loader.TwoCropsTransform(train_transforms1, train_transforms2),
                                use_z_score=args.use_z_score, 
                                return_original=False,
                                use_label=True)
    elif args.name_dataset == "brats-160":
        
        dataset_train = brats_160.Brats160Data( 
                                                transform=moco.loader.TwoCropsTransform(train_transforms1, train_transforms2),
                                                mode = 'train',
                                                use_z_score=args.use_z_score,
                                                pre_load=False
                                                )
    elif args.name_dataset == "ct-176":
        dataset_train = ct_176.CT176Data(
            mode = 'train',
            transform=moco.loader.TwoCropsTransform(train_transforms1, train_transforms2),
            use_z_score=args.use_z_score,
            pre_load=False
        )
    else:
        raise ValueError("Unknown 3D dataset")
    
    print(dataset_train)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, optimizer, scaler, summary_writer, epoch, args)

        if (not args.multiprocessing_distributed and (epoch + 1) % args.save_freq == 0) or (args.multiprocessing_distributed
                and args.rank == 0 and (epoch + 1) % args.save_freq == 0): # only the first GPU saves checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scaler': scaler.state_dict(),
            }, filename='checkpoint_%04d.pth.tar' % epoch, args=args)

    if args.rank == 0:
        summary_writer.close()
    

def put_model_for_distributed_training(args, model, ngpus_per_node):
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    return model


def create_model_and_optim(args, ngpus_per_node):
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'resnet_3d':
        model = moco.builder.MoCo_ResNet(
            partial(generate_model, model_depth=18, num_classes=args.num_classes, n_input_channels=args.in_channels),
            args.moco_dim, args.moco_mlp_dim, args.moco_t)
        print("resnet3d model created")
    elif args.arch == "vim_3d":
        #  vim_3D_small_patch16_stride16_224_bimambav2_final_pool_mean_abs_pos_embed_div2_for_mocov3
        model = moco.builder.MoCo_ViT(
            partial(vim_3D_small_patch16_stride16_224_bimambav2_final_pool_mean_abs_pos_embed_div2, 
                    channels=args.in_channels, img_size=args.volume_size),
            args.moco_dim, args.moco_mlp_dim, args.moco_t)
    elif args.arch == "vim_mha_3d":
         model = moco.builder.MoCo_ViT(
            partial(vim_mha_3D_small_patch16_stride16_final_pool_mean_abs_pos_embed_div2, 
                    channels=args.in_channels, img_size=args.volume_size),
            args.moco_dim, args.moco_mlp_dim, args.moco_t)
    elif args.arch == "vim2_3d":
         model = moco.builder.MoCo_ViT(
            partial(vim2_3D_small_patch16_stride16_final_pool_mean_abs_pos_embed_div2, 
                    channels=args.in_channels, img_size=args.volume_size),
            args.moco_dim, args.moco_mlp_dim, args.moco_t)
    else:
        model = moco.builder.MoCo_ViT(
            partial(VisionTransformer3D, in_chans=args.in_channels, volume_size=args.volume_size),
            args.moco_dim, args.moco_mlp_dim, args.moco_t)
        print("3d vision transformer model created")
    
    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size / 256
    model = put_model_for_distributed_training(args, model, ngpus_per_node)
    print(model)  # print model after SyncBatchNorm
    if args.optimizer == 'lars':
        optimizer = moco.optimizer.LARS(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                      weight_decay=args.weight_decay)
    return model, optimizer


def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        # print('images length', len(images))
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(True):
            loss = model(images[0], images[1], moco_m)

        losses.update(loss.item(), images[0].size(0))
        if args.rank == 0:
            summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    # Let us return the avrage loss
    return losses


def save_checkpoint(state, filename='checkpoint.pth.tar', args=None):
    os.makedirs(args.dump_path, exist_ok=True)
    filename = os.path.join(args.dump_path, filename)
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (
                1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_all_feat_und_labels(dataset_whole, args):
    mri, labels = [], []
    for idx in range(len(dataset_whole)):
        mri.append(dataset_whole[idx][0])
        labels.append(dataset_whole[idx][-1])
    if args.in_channels == 1:
        return torch.cat(mri), torch.stack(labels)
    else:
        return torch.stack(mri), torch.stack(labels)


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


if __name__ == '__main__':
    main()