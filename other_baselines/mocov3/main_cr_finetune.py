#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
import argparse
import numpy as np
import os

import json
import datetime

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from sklearn.model_selection import KFold
from timm.data.mixup import Mixup
import torch.utils.data
import torch.utils.data.distributed

from torch.utils.tensorboard import SummaryWriter
import sys
import time
sys.path.append('../../')

from utils import brats, brats_160, medmnist
from utils.datasets import build_dataset
import utils.misc as misc
from utils.pos_embed import interpolate_pos_embed

from timm.models.layers import trunc_normal_
import utils.lr_decay as lrd
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from other_baselines.mocov3.moco.resent3d_base import generate_model
from other_baselines.mocov3.moco.vit_3d import VisionTransformer3D
from timm.utils import ModelEma
import torchio as tio
import torchvision.transforms as transforms
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_finetune import train_one_epoch, evaluate, evaluate_3d, get_global_metrics
from models_vim import vim2_3D_small_patch16_stride16_final_pool_mean_abs_pos_embed_div2, vim_3D_small_patch16_stride16_224_bimambav2_final_pool_mean_abs_pos_embed_div2

model_names = ['vit_3d', 'resnet_3d', 'vim_3d', "vim2_3d"]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# The dataset we are going to use
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_3d',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--accum_iter', default=1, type=int,
                    help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')


parser.add_argument('--input_size', default=224, type=int,
                    help='images input size')

parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')

# model ema parameters
parser.add_argument('--model-ema', action='store_true')
parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
parser.set_defaults(model_ema=True)
parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

# Optimizer parameters
parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')

parser.add_argument('--lr', type=float, default=None, metavar='LR',
                    help='learning rate (absolute lr)')
parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                    help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
parser.add_argument('--layer_decay', type=float, default=0.75,
                    help='layer-wise lr decay from ELECTRA/BEiT')

parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0')

parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR')

# Augmentation parameters
parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                    help='Color jitter factor (enabled only when not using Auto/RandAug)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')

# * Random Erase params
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')

# * Mixup params
parser.add_argument('--mixup', type=float, default=0,
                    help='mixup alpha, mixup enabled if > 0.')
parser.add_argument('--cutmix', type=float, default=0,
                    help='cutmix alpha, cutmix enabled if > 0.')
parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup_prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup_mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

parser.add_argument('--global_pool', action='store_true')
parser.set_defaults(global_pool=True)
parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                    help='Use class token instead of global pool for classification')
parser.add_argument('--use_scheduler', action='store_true',)
parser.set_defaults(use_scheduler=False)

# Dataset parameters
parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                    help='dataset path')
parser.add_argument('--in_channels', type=int, default=4)
# parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--num_classes', default=1000, type=int,
                    help='number of the classification types')
parser.add_argument('--use_3d', action='store_true', help='use 3D data')
parser.add_argument("--name_dataset", default='brats', type=str,)
parser.add_argument("--k_fold", default= 3 , type = int, help = 'the k value of cross validation')
parser.add_argument('--use_z_score', action='store_true', help='use z-score normalization for 3D data')
parser.add_argument('--label_type', default= 'IDH', type = str, help = "the label type of egd dataset")

parser.add_argument('--output_dir', default='./output_dir',
                    help='path where to save, empty for no saving')
parser.add_argument('--log_dir', default='./output_dir',
                    help='path where to tensorboard log')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--resume', default='',
                    help='resume from checkpoint')

parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--dist_eval', action='store_true', default=False,
                    help='Enabling distributed evaluation (recommended during training for faster monitor')
parser.add_argument('--num_workers', default=10, type=int)
parser.add_argument('--pin_mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
parser.set_defaults(pin_mem=True)

# distributed training parameters
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')
# additional configs:
parser.add_argument('--pretrained', type=str,
                    help='path to moco pretrained checkpoint',
                    default='vit_egd_data/min_loss.pth.tar')

def main():
    args = parser.parse_args()

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    if args.use_3d:
        transforms_ = [
        tio.RandomAffine(),
        tio.RandomNoise(std=0.1),
        tio.RandomGamma(log_gamma=(-0.3, 0.3))
        ]
    
        train_transforms = tio.Compose(transforms_)

        if args.name_dataset == "brats-160":
            
            whole_dataset = brats_160.Brats160Data( 
                                                transform= None,
                                                mode = 'test',
                                                label_type= args.label_type,
                                                use_z_score=args.use_z_score,
                                                pre_load=False
                                                )
        else:
            raise ValueError("Unknown 3D dataset")
    else:
        raise ValueError("Unknown 2D dataset")
    
    distribute_flag = True
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
    else:
        distribute_flag = False 
    
    kfold = KFold(n_splits=args.k_fold, shuffle=True)
    
    results = {}
    out_gt = torch.FloatTensor().to('cpu')
    out_pred= torch.FloatTensor().to('cpu')
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(whole_dataset)):
        print(f"Start finetuning for fold_{fold}")
        dataset_train = torch.utils.data.Subset(whole_dataset, train_ids)
        dataset_train.transform = transforms_
        dataset_val = torch.utils.data.Subset(whole_dataset, test_ids)
        
        
        dataset_train.transform = train_transforms
        if distribute_flag:
            sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
            if args.dist_eval:
                if len(dataset_val) % num_tasks != 0:
                    print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            
        if global_rank == 0 and args.log_dir is not None and not args.eval:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.log_dir)
        else:
            log_writer = None

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            print("Mixup is activated!")
            
            mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.num_classes)
            
        # create model
          #  # create model
        print("=> creating model '{}'".format(args.arch))
        if args.arch == 'resnet_3d':
            model = generate_model(model_depth=18, num_classes=args.num_classes, n_input_channels=args.in_channels)
            print("resnet3d model created")
            linear_keyword = 'fc'
            
        elif args.arch == "vim_3d":
        
            model = vim_3D_small_patch16_stride16_224_bimambav2_final_pool_mean_abs_pos_embed_div2(
                img_size = args.input_size, num_classes = args.num_classes, 
                channels = args.in_channels)
            linear_keyword = "head"
        elif args.arch == "vim2_3d":
            model = vim2_3D_small_patch16_stride16_final_pool_mean_abs_pos_embed_div2( 
                img_size = args.input_size, 
                num_classes = args.num_classes, 
                channels = args.in_channels)
            linear_keyword = "head"
        else:
            # A negative num-classes ensures that the classifier layer is not created and rather, replaced with the Identity
            model = VisionTransformer3D(in_chans=args.in_channels, volume_size=args.input_size, num_classes=args.num_classes)
            print("3d vision transformer model created")
            linear_keyword = 'head'
            
        if not args.pretrained:
            raise AttributeError("Please specify a pretrained model checkpoint.")
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']

            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('base_encoder') and not k.startswith('base_encoder.%s' % linear_keyword):
                    # remove prefix
                    state_dict[k[len("base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            interpolate_pos_embed(model, state_dict)
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
            # assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
            if 'vit' in args.arch:
                trunc_normal_(model.head.weight, std=2e-5)
    
        model.to(device)
        
        model_ema = None
        if args.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            model_ema = ModelEma(
                model,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else '',
                resume='')
            print("Using model ema")
        
        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # print("Model = %s" % str(model_without_ddp))
        print('number of params (M): %.2f' % (n_parameters / 1.e6))

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
        
        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256

        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        # build optimizer with layer-wise lr decay (lrd)
        try:
            param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
            no_weight_decay_list=model_without_ddp.no_weight_decay(),
            layer_decay=args.layer_decay
            )
            optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        except:
            optimizer = torch.optim.AdamW(model_without_ddp.parameters(), args.lr,
                                      weight_decay=args.weight_decay)
        loss_scaler = NativeScaler()

        if mixup_fn is not None:
            # smoothing is handled with mixup label transform
            # if args.name_dataset == "brats":
            #     criterion = SoftCrossEntropyWithWeightsLoss(weights=args.cross_entropy_wt)
            # else:
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing > 0.:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            # if args.name_dataset == "brats":
            #     criterion = torch.nn.CrossEntropyLoss(weight=args.cross_entropy_wt)
            # else:
            criterion = torch.nn.CrossEntropyLoss()

        print("criterion = %s" % str(criterion))

        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, model_ema = model_ema)

        if args.eval:
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            exit(0)

        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        max_accuracy = 0.0
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, mixup_fn,
                log_writer=log_writer,
                args=args, model_ema=model_ema
            )
            if args.output_dir and epoch == args.epochs - 1:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, post_name= f"_fold_{fold}_epoch_{epoch}",
                    model_ema = model_ema)
            
            if args.use_3d:
                if epoch == args.epochs - 1:
                    test_stats, gt, pred = evaluate_3d(data_loader_val, model, device, return_gt_pred= True)
                    out_gt = torch.cat((out_gt, gt.to('cpu')), 0)
                    out_pred = torch.cat((out_pred, pred.to('cpu')), 0)
                else: 
                    test_stats = evaluate_3d(data_loader_val, model, device)
                    
                print(
                    f"ROC_AUC score of the network on the {len(sampler_val)} val images: {test_stats['roc_auc_score']:.1f}%")
                if log_writer is not None:
                    # Writing the logs
                    log_writer.add_scalar('ft/roc_auc_score', test_stats['roc_auc_score'], epoch)
                    log_writer.add_scalar('ft/loss', test_stats['loss'], epoch)
                    # log_writer.add_scalar('ft/roc_auc_score', test_stats['roc_auc_score'], epoch)
                    # log_writer.add_scalar('ft/loss', test_stats['loss'], epoch)
 
            else:
                test_stats = evaluate(data_loader_val, model, device)
                print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
                max_accuracy = max(max_accuracy, test_stats["acc1"])
                print(f'Max accuracy: {max_accuracy:.2f}%')

                if log_writer is not None:
                    log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                    log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
                    log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                **{f'test_{k}': v for k, v in test_stats.items()},
                                'fold': fold,
                                'epoch': epoch,
                                'n_parameters': n_parameters}

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
                if args.use_3d and epoch == args.epochs -1 :
                    results[fold] = test_stats
    
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('fold_{} Training time {}'.format(fold, total_time_str))
        print()
    
    if args.output_dir and misc.is_main_process():
        key_list = list(results[0].keys())
        key_dict = {}
        for key in key_list:
            key_dict[key] = 0
            for f in results.keys():
                value = results[f][key]
                key_dict[key] += value
        print("final avg results")
        for key in key_dict.keys():
            print(key, key_dict[key] / args.k_fold)
            key_dict[key] = key_dict[key] / args.k_fold
        global_test_stats = get_global_metrics(out_gt, out_pred)
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(key_dict) + "\n")

        global_dict = {}
        print()
        print("final global results")
        for key in global_test_stats.keys():
            print(key, global_test_stats[key])
            global_dict[key] = global_test_stats[key]
        
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(global_dict) + "\n")

if __name__ == '__main__':
    main()
