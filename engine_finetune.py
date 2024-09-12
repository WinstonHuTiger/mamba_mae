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
from typing import Iterable, Optional
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch
import numpy as np 

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score as f1_score_sk

# Need to convert the values in one-hot encoding
enc = OneHotEncoder()
possible_labels = np.array([0, 1]).reshape(-1, 1)
enc.fit(possible_labels)

from timm.data import Mixup
from timm.utils import accuracy

import utils.misc as misc
import utils.lr_sched as lr_sched

def find_vals(predictions, target):
    predictions = torch.max(predictions, dim=1)[1]  # We need the indices for the max

    
    cm = confusion_matrix(predictions.cpu().numpy(), target.cpu().numpy())
    print('predictions:', predictions)
    print('targets:', target)
    print('confusion matrix', cm)
    specificity = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    print('Combined:')
    print('specificity:', specificity)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    print('sensitivity:', sensitivity)
    return specificity, sensitivity



def roc_auc(predictions, target):
    # Converting raw scores into probabilities
    specificity, sensitivity = find_vals(predictions, target)
    predictions = torch.softmax(predictions, dim=1)
    predictions, target = predictions.cpu().numpy(), target.cpu().numpy()
    target_one_hot = enc.transform(target.reshape(-1, 1)).toarray()  # Reshaping needed by the library
    # Arguments take 'GT' before taking 'predictions'
    return roc_auc_score(target_one_hot, predictions), specificity, sensitivity

def f1_score(y_true, y_pred, is_binary=True, epsilon=1e-7):
    """
    Calculate the F1 score between tensors of true labels and predicted labels.

    Args:
    - y_true (Tensor): True labels.
    - y_pred (Tensor): Predicted labels or logits. If not binary, should be softmax probabilities.
    - is_binary (bool): Set to True if the task is binary classification, else False.
    - epsilon (float): Small value to avoid division by zero.

    Returns:
    - f1 (Tensor): The F1 score.
    """
    y_pred = torch.softmax(y_pred, dim=1)
    y_pred = torch.max(y_pred, dim=1)[1]
    if not is_binary:
        y_pred = y_pred.argmax(dim=1)
    
    true_positives = torch.sum(y_pred * y_true)
    predicted_positives = torch.sum(y_pred)
    possible_positives = torch.sum(y_true)

    precision = true_positives / (predicted_positives + epsilon)
    recall = true_positives / (possible_positives + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    return f1

@torch.no_grad()
def generate_features(data_loader, model, device):
    """
    We save labels and predictions together. Saving labels just makes our life easier in terms of having pickles stored at the same location
    :param data_loader: train/test
    :param model: vit model
    :param device: cuda/cpu
    :param ssl_feature_dir: location for storing features
    :param feature_file_name: npy file for features default: features.npy
    :param label_file_name: npy file for labels default: gt_labels.npy
    :param log_writer: SummaryWriter default:None
    :return: None
    """
    # switch to evaluation mode
    model.eval()
    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)
    with torch.no_grad():
        for batch in data_loader:
            images = batch[0]
            target = batch[-1]
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model.forward_features(images)
            outPRED = torch.cat((outPRED, output), 0)
            outGT = torch.cat((outGT, target), 0)
    return outPRED, outGT

@torch.no_grad()
def knn_evaluate(train_features, train_labels, val_features, val_labels, device, log_writer = None, k=3):
    metric_logger = misc.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    metric_logger.add_meter('knn_acc', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    train_features = train_features.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    train_labels = train_labels.to(device)
    m = train_features.size(0)
    n = val_features.size(0)
    train_features = train_features.view(m, -1)
    val_features = val_features.view(n, -1)
    xx = torch.pow(train_features, 2).sum(dim=1, keepdim=True).expand(m, n) 
    yy = torch.pow(val_features, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.mm(train_features, val_features.t())
    # dist, indices = dist.topk(k, largest=False, sorted=False)
    # indices = train_labels[indices]
    # pred = indices.mode(dim=1).values
    
    num_train, num_test = dist.shape
    y_pred = torch.zeros(num_test, dtype=torch.int64).to(device)
    ##############################################################################
    # TODO: Implement this function. You may use an explicit loop over the test  #
    # samples. Hint: Look up the function torch.topk                             #
    ##############################################################################
    # Replace "pass" statement with your code
    for i in range(num_test):
        values, indices = torch.topk(dist[:, i], k, largest=False)
        values, indices = train_labels[indices].unique(return_counts=True)
        y_pred[i] = values[indices.argmax()]
    
    assert y_pred.size(0) == val_labels.size(0)
    acc = (y_pred == val_labels).float().mean()
    # predictions, target = y_pred.cpu().numpy(), val_labels.cpu().numpy()
    # target_one_hot = enc.transform(target.reshape(-1, 1)).toarray()
    targets = val_labels.cpu().numpy()
    preds = y_pred.cpu().numpy()
    auc = roc_auc_score(targets, preds)
    # auc, specificity, sensitivity = roc_auc(y_pred, val_labels)
    f1_ = f1_score_sk(targets, preds)
    metric_logger.update(knn_acc=acc.item())
    metric_logger.update(knn_auc = auc)
    metric_logger.update(knn_f1 = f1_)
    if log_writer is not None:
        log_writer.add_scalar('knn_acc', acc, 0)
        log_writer.add_scalar("knn_auc", auc, 0)
        log_writer.add_scalar("knn_f1", f1_, 0)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, y_pred
        
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None, model_ema = None ):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # criterion = criterion.to(device)
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0 and args.use_scheduler:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        # print(targets.shape)
        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(samples)
            # print(outputs.shape)
            # print(outputs.shape)
            # print(targets.shape)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        
        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device ):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_3d(data_loader, model, device, return_gt_pred=False):
    # Weights for breast_tumor = 2:1 majority being label 0
    # Since evaluation is always hard target and not SoftTarget
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=False):
            output = model(images)
            loss = criterion(output, target)
        acc1 = accuracy(output, target)[0]
        outPRED = torch.cat((outPRED, output), 0)
        outGT = torch.cat((outGT, target), 0)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)   
        # metric_logger.meters['specificity'].update(specificity, n=batch_size)
        # metric_logger.meters['sensitivity'].update(sensitivity, n=batch_size)
    roc_auc_score, specificity, sensitivity = roc_auc(predictions=outPRED, target=outGT)
    f1_ = f1_score(outGT, outPRED).item()
    metric_logger.update(roc_auc_score=roc_auc_score)
    metric_logger.update(f1_score = f1_)
    
    metric_logger.update(specificity=specificity)
    metric_logger.update(sensitivity=sensitivity)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* roc_auc_score {:.3f}, F1-score {:.3f}, Acc@1 {top1.global_avg:.3f} ,loss {losses.global_avg:.3f}'
          .format(roc_auc_score, f1_,  top1=metric_logger.acc1, losses=metric_logger.loss))
    if return_gt_pred:
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, outGT, outPRED
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def get_global_metrics(gt, pred):
    assert gt.shape[0] == pred.shape[0]
    
    metric_logger = misc.MetricLogger(delimiter="  ")
    acc1 = accuracy(pred, gt)[0]
    f1_ = f1_score(gt, pred).item()
    roc_auc_score, specificity, sensitivity = roc_auc(predictions=pred, target=gt)
    metric_logger.update(roc_auc_score=roc_auc_score)
    metric_logger.update(f1_score = f1_)
    metric_logger.update(specificity=specificity)
    metric_logger.update(sensitivity=sensitivity)
    metric_logger.update(acc1=acc1.item())
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    