import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# import timm

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import torchio as tio
from utils import medmnist
from utils import brats
import utils.misc as misc
from utils.pos_embed import interpolate_pos_embed
from utils import brats_160
from utils.crop import RandomResizedCrop

import models_vit
import models_vim

from sklearn.manifold import TSNE   
import matplotlib.pyplot as plt

from engine_finetune import train_one_epoch, evaluate, evaluate_3d, generate_features, knn_evaluate

def get_args_parser():
    parser = argparse.ArgumentParser('MAE KNN for features evaluation', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    
    
    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--use_3d', action='store_true', help='use 3D data')
    parser.add_argument("--name_dataset", default='brats', type=str,)
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
    
    parser.add_argument('--num_k', default=5, type=int, help='number of neighbors for KNN')
                        
                        

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

    return parser

def visualization_tsne(val_features, y_pred, save_dir):
    # 假设 X 是特征数据，labels 是KNN的聚类标签
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(val_features)

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_pred, cmap='viridis')
    plt.colorbar()
    plt.title('t-SNE Visualization of KNN Clustering')
    plt.savefig(os.path.join(save_dir, 'knn_tsne.png'))
    

def main(args):
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
        # transforms_ = [
        # tio.RandomAffine(),
        # tio.RandomNoise(std=0.1),
        # tio.RandomGamma(log_gamma=(-0.3, 0.3))
        # ]
    
        # train_transforms = tio.Compose(transforms_)

        if args.name_dataset == 'brats':
            dataset_train = brats.build_dataset( mode='train',
                                                
                                                transforms=None,
                                    use_z_score=args.use_z_score, 
                                    return_original=False,
                                    use_label=True)
            dataset_val = brats.build_dataset( mode='val',
                                              transforms=None,
                                              use_z_score=args.use_z_score,
                                              return_original=False,
                                              use_label=True)
            print("Using 3, 1 weight for Brats")
            args.cross_entropy_wt = torch.as_tensor([3.0, 1.0])
        elif args.name_dataset == "brats-160":
            
            whole_dataset = brats_160.Brats160Data( 
                                                transform= None,
                                                mode = 'test',
                                                label_type= args.label_type,
                                                use_z_score=args.use_z_score,
                                                pre_load=False
                                                )
            length = len(whole_dataset)
            train_size= int(0.8 * length)
            val_size = length - train_size 
            generator1 = torch.Generator().manual_seed(seed)
            dataset_train, dataset_val  = torch.utils.data.random_split(whole_dataset, [train_size, val_size], generator=generator1)
            # dataset_train.transform = train_transforms
        else:
            raise ValueError("Unknown 3D dataset")

    else:
        if 'mnist'in args.name_dataset :
            data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
            dataset_train = medmnist.build_dataset(data_flag=args.name_dataset, mode='train', 
                                                   
                                                   transforms = data_transform, size = args.input_size)
            dataset_val = medmnist.build_dataset(data_flag=args.name_dataset, 
                                                 mode='test',
                                                 transforms = data_transform,
                                                 size = args.input_size)
        else:
            # linear probe: weak augmentation
            transform_train = transforms.Compose([
                    RandomResizedCrop(224, interpolation=3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            transform_val = transforms.Compose([
                    transforms.Resize(256, interpolation=3),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
            dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)
    print(dataset_train)
    print(dataset_val)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
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

    num_channels = 3
    if args.use_3d:
        num_channels = 1
        if args.name_dataset == 'egd':
            num_channels = 4
        if args.name_dataset == 'brats-160':
            num_channels = 4
        if args.name_dataset == 'ct-176':
            num_channels = 1
    else:
        num_channels = 3
        if 'mnist' in args.name_dataset:
            num_channels = dataset_train.info['n_channels']
    
    
    if "vim" in args.model :
        model = models_vim.__dict__[args.model](
            img_size = args.input_size,
            num_classes=args.nb_classes, 
            drop_path_rate=args.drop_path,
            channels = num_channels,
                                )
    
    else:
        model = models_vit.__dict__[args.model](
            img_size = args.input_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        in_chans = num_channels
        )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # if args.global_pool:
        #     if 'vim' in args.model:
        #         assert set(msg.missing_keys) == {'rope.freqs_cos', 'rope.freqs_sin', 'head.weight', 'head.bias'}
        #     else:
        #         assert set(msg.missing_keys) == { 'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        #     if 'vim' in args.model:
        #         assert set(msg.missing_keys) == {'rope.freqs_cos', 'rope.freqs_sin', 'head.weight', 'head.bias'}
        #     else:
        #         assert set(msg.missing_keys) == {'rope.freqs_cos', 'rope.freqs_sin','head.weight', 'head.bias'}


    #     # manually initialize fc layer: following MoCo v3
    #     trunc_normal_(model.head.weight, std=0.01)

    # # for linear prob only
    # # hack: revise model's head with BN
    # model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # # freeze all
    for _, p in model.named_parameters():
        p.requires_grad = False
    # for _, p in model.head.named_parameters():
    #     p.requires_grad = True

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * 1 * misc.get_world_size()
    
    # if args.lr is None:  # only base_lr is specified
    #     args.lr = args.blr * eff_batch_size / 256

    # print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    # print("actual lr: %.2e" % args.lr)

    # print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # print(optimizer)
    # loss_scaler = NativeScaler()

    # criterion = torch.nn.CrossEntropyLoss()

    # print("criterion = %s" % str(criterion))

    misc.load_model_for_features(args=args, model_without_ddp=model_without_ddp)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start knn evalutation for", args.model)
    start_time = time.time()
    # max_accuracy = 0.0
    # for epoch in range(args.start_epoch, args.epochs):
    if args.distributed:
        data_loader_train.sampler.set_epoch(0)
        
    features, gt = generate_features(data_loader_train, model, device)
    
    if args.output_dir:
        
        features = misc.concat_all_gather(features)
        gt = misc.concat_all_gather(gt)
        misc.save_features( features, file_name = 'features_train.npy', output_dir= args.output_dir)
        misc.save_features( gt, file_name = 'gt_labels_train.npy', output_dir= args.output_dir)
        

    
    features_val, gt_val = generate_features(data_loader_val, model, device)
    if args.output_dir:
        features_val = misc.concat_all_gather(features_val)
        gt_val = misc.concat_all_gather(gt_val)
        misc.save_features( features_val, file_name = 'features_val.npy', output_dir= args.output_dir)
        misc.save_features( gt_val, file_name = 'gt_labels_val.npy', output_dir= args.output_dir)
        
    if misc.is_main_process():
        test_stats, y_pred = knn_evaluate(features, gt, features_val, gt_val, device, k = args.num_k)
        features_val = features_val.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        visualization_tsne(features_val, y_pred, args.output_dir)
    
    if log_writer is not None:
        log_writer.add_scalar('perf/knn_acc', test_stats['knn_acc'], 0)
    log_stats = {
                    **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': 0,
                        'n_parameters': n_parameters}

    if args.output_dir and misc.is_main_process():
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Feature Generation and evaluation time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    
    