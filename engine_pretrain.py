# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import utils.misc as misc
import utils.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=False):
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_stage_epoch(model: torch.nn.Module,
                          data_loader: Iterable, optimizer: torch.optim.Optimizer,
                          device: torch.device, epoch: int, loss_scaler,
                          log_writer=None,
                          args=None,
                          edge_map_weight=0):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    criterion = torch.nn.CosineSimilarity(dim=1).to(device)
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (sample, original_volume, _) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        sample = sample.to(device, non_blocking=True)
        original_volume = original_volume.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=False):
            loss, pred, mask, p1, p2, z1, z2 = model(view1=sample, view2=original_volume, mask_ratio=args.mask_ratio,
                                                     edge_map_weight=edge_map_weight)

        # contr_loss = compute_contrastive_loss(args, criterion, p1, p2, z1, z2)

        # This is based on our modification for weighted loss
        # loss_value = loss.item()
        weighted_loss, edge_map_loss, reconstruction_loss, perceptual_loss = loss[0], loss[1], loss[2], loss[3]
        # loss = loss[0] 
        loss_value = loss.item()
        metric_logger.update(edge_map_loss=edge_map_loss)
        metric_logger.update(reconstruction_loss=reconstruction_loss)
        metric_logger.update(perceptual_loss=perceptual_loss)
        # metric_logger.update(contr_loss=contr_loss)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        #  Our extra logs
        reconstruction_loss_value_reduce = misc.all_reduce_mean(reconstruction_loss)
        edge_map_loss_value_reduce = misc.all_reduce_mean(edge_map_loss)
        perceptual_loss_loss_value_reduce = misc.all_reduce_mean(perceptual_loss)
        # contr_loss_value_reduce = misc.all_reduce_mean(contr_loss)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

            log_writer.add_scalar('reconstruction_loss', reconstruction_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('sobel_loss', edge_map_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('perceptual_loss', perceptual_loss_loss_value_reduce, epoch_1000x)
            # log_writer.add_scalar('contr_loss', contr_loss_value_reduce, epoch_1000x)
        # Deleting the variables. Perhaps this allows us to use memory better
        del sample
        del original_volume
        torch.cuda.empty_cache()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

