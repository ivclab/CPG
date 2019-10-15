"""Main entry point for doing all stuff."""
# always choose the best checkpoint

from __future__ import division, print_function

import argparse
import json
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn.parameter import Parameter

import UTILS.utils as utils
import pdb
import os
import math
from tqdm import tqdm
import sys
import numpy as np
from pprint import pprint

import models.layers as nl
import models
from UTILS.manager import Manager
import UTILS.dataset as dataset
import logging

# To prevent PIL warnings.
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='resnet50',
                   help='Architectures')
parser.add_argument('--num_classes', type=int, default=-1,
                   help='Num outputs for dataset')
# Optimization options.
parser.add_argument('--lr', type=float, default=0.1,
                   help='Learning rate for parameters, used for baselines')
parser.add_argument('--lr_mask', type=float, default=1e-4,
                   help='Learning rate for mask')
parser.add_argument('--lr_mask_decay_every', type=int,
                   help='Step decay every this many epochs')

# parser.add_argument('--lr_classifier', type=float,
#                    help='Learning rate for classifier')
# parser.add_argument('--lr_classifier_decay_every', type=int,
#                    help='Step decay every this many epochs')

parser.add_argument('--batch_size', type=int, default=32,
                   help='input batch size for training')
parser.add_argument('--val_batch_size', type=int, default=100,
                   help='input batch size for validation')
parser.add_argument('--workers', type=int, default=24, help='')
parser.add_argument('--weight_decay', type=float, default=0.0,
                   help='Weight decay')
# Masking options.
parser.add_argument('--mask_init', default='1s',
                   choices=['1s', 'uniform', 'weight_based_1s'],
                   help='Type of mask init')
parser.add_argument('--mask_scale', type=float, default=1e-2,
                   help='Mask initialization scaling')
parser.add_argument('--mask_scale_gradients', type=str, default='none',
                   choices=['none', 'average', 'individual'],
                   help='Scale mask gradients by weights')
parser.add_argument('--threshold_fn',
                   choices=['binarizer', 'ternarizer'],
                   help='Type of thresholding function')
parser.add_argument('--threshold', type=float, default=2e-3, help='')
# Paths.
parser.add_argument('--dataset', type=str, default='',
                   help='Name of dataset')
parser.add_argument('--train_path', type=str, default='',
                   help='Location of train data')
parser.add_argument('--val_path', type=str, default='',
                   help='Location of test data')
parser.add_argument('--save_prefix', type=str, default='checkpoints/',
                   help='Location to save model')
# Other.
parser.add_argument('--cuda', action='store_true', default=True,
                   help='use CUDA')
# parser.add_argument('--no_mask', action='store_true', default=False,
#                    help='Used for running baselines, does not use any masking')

parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--checkpoint_format', type=str, 
    default='./{save_folder}/checkpoint-{epoch}.pth.tar',
    help='checkpoint file format')

parser.add_argument('--epochs', type=int, default=160,
                    help='number of epochs to train')
parser.add_argument('--restore_epoch', type=int, default=0, help='')
parser.add_argument('--image_size', type=int, default=32, help='')
parser.add_argument('--save_folder', type=str,
                    help='folder name inside one_check folder')
parser.add_argument('--load_folder', default='', help='')

# parser.add_argument('--datadir', default='/home/ivclab/decathlon-1.0/', 
#                    help='folder containing data folder')
# parser.add_argument('--imdbdir', default='/home/ivclab/decathlon-1.0/annotations', 
#                    help='annotation folder')

# parser.add_argument('--train_weight', action='store_true', default=False, help='')
# parser.add_argument('--train_mask', action='store_true', default=False, help='')
# parser.add_argument('--train_classifier', action='store_true', default=False, help='')

parser.add_argument('--pruning_interval', type=int, default=100, help='')
parser.add_argument('--pruning_frequency', type=int, default=10, help='')
parser.add_argument('--initial_sparsity', type=float, default=0.0, help='')
parser.add_argument('--target_sparsity', type=float, default=0.1, help='')

parser.add_argument('--mode',
                   choices=['finetune', 'prune', 'inference'],
                   help='Run mode')

parser.add_argument('--jsonfile', type=str, help='file to restore baseline validation accuracy')
parser.add_argument('--network_width_multiplier', type=float, default=1.0, help='the multiplier to scale up the channel width')
# parser.add_argument('--tmp_benchmark_file', type=str, default='tmp_benchmark_file.txt', help='')
parser.add_argument('--test_piggymask', action='store_true', default=False, help='')
class Optimizers(object):
    def __init__(self):
        self.optimizers = []
        self.lrs = []
        # self.args = args

    def add(self, optimizer, lr):
        self.optimizers.append(optimizer)
        self.lrs.append(lr)

    def step(self):
        for optimizer in self.optimizers:
            # if isinstance(optimizer, torch.optim.Adam):
                # pdb.set_trace()
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def __getitem__(self, index):
        return self.optimizers[index]

    def __setitem__(self, index, value):
        self.optimizers[index] = value

def main():
    """Do stuff."""
    args = parser.parse_args()
    # don't use this, neither set learning rate as a linear function
    # of the count of gpus, it will make accuracy lower
    # args.batch_size = args.batch_size * torch.cuda.device_count()
    args.network_width_multiplier = math.sqrt(args.network_width_multiplier)

    if args.save_folder and not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        args.cuda = False
        
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    resume_folder = args.load_folder
    for try_epoch in range(200, 0, -1):
        if os.path.exists(args.checkpoint_format.format(
            save_folder=resume_folder, epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break

    if args.restore_epoch:
        resume_from_epoch = args.restore_epoch

    # Set default train and test path if not provided as input.
    utils.set_dataset_paths(args)
                        
    if resume_from_epoch:
        filepath = args.checkpoint_format.format(save_folder=resume_folder, epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        checkpoint_keys = checkpoint.keys()        
        dataset_history = checkpoint['dataset_history']
        dataset2num_classes = checkpoint['dataset2num_classes']
        masks = checkpoint['masks']
        shared_layer_info = checkpoint['shared_layer_info']

        if 'num_for_construct' in checkpoint_keys:
            num_for_construct = checkpoint['num_for_construct']
        if args.mode == 'inference' and 'network_width_multiplier' in shared_layer_info[args.dataset]:
            args.network_width_multiplier = shared_layer_info[args.dataset]['network_width_multiplier']
    else:
        dataset_history = []
        dataset2num_classes = {}
        masks = {}
        shared_layer_info = {}

    if args.arch == 'resnet50':
        num_for_construct = [64, 64, 64*4, 128, 128*4, 256, 256*4, 512, 512*4]
        model = models.__dict__[args.arch](pretrained=True, num_for_construct=num_for_construct, threshold=args.threshold)
    elif 'vgg' in args.arch:
        custom_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        model = models.__dict__[args.arch](custom_cfg, dataset_history=dataset_history, dataset2num_classes=dataset2num_classes, 
            network_width_multiplier=args.network_width_multiplier, shared_layer_info=shared_layer_info)
    else:
        print('Error!')
        sys.exit(1)

    # Add and set the model dataset.
    model.add_dataset(args.dataset, args.num_classes)
    model.set_dataset(args.dataset)

    model = nn.DataParallel(model)
    model = model.cuda()

    if not masks:
        for name, module in model.named_modules():
            if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                mask = torch.ByteTensor(module.weight.data.size()).fill_(0)
                if 'cuda' in module.weight.data.type():
                    mask = mask.cuda()
                masks[name] = mask
    else:
        # when we expand network, we need to allocate new masks
        NEED_ADJUST_MASK = False
        for name, module in model.named_modules():
            if isinstance(module, nl.SharableConv2d):
                if masks[name].size(1) < module.weight.data.size(1):
                    assert args.mode == 'finetune'
                    NEED_ADJUST_MASK = True
                elif masks[name].size(1) > module.weight.data.size(1):
                    assert args.mode == 'inference'
                    NEED_ADJUST_MASK = True
                

        if NEED_ADJUST_MASK:
            if args.mode == 'finetune':
                for name, module in model.named_modules():
                    if isinstance(module, nl.SharableConv2d):
                        mask = torch.ByteTensor(module.weight.data.size()).fill_(0)
                        if 'cuda' in module.weight.data.type():
                            mask = mask.cuda()
                        mask[:masks[name].size(0), :masks[name].size(1), :, :].copy_(masks[name])
                        masks[name] = mask
                    elif isinstance(module, nl.SharableLinear):
                        mask = torch.ByteTensor(module.weight.data.size()).fill_(0)                
                        if 'cuda' in module.weight.data.type():
                            mask = mask.cuda()
                        mask[:masks[name].size(0), :masks[name].size(1)].copy_(masks[name])
                        masks[name] = mask
            elif args.mode == 'inference':            
                for name, module in model.named_modules():
                    if isinstance(module, nl.SharableConv2d):
                        mask = torch.ByteTensor(module.weight.data.size()).fill_(0)
                        if 'cuda' in module.weight.data.type():
                            mask = mask.cuda()
                        mask[:, :, :, :].copy_(masks[name][:mask.size(0), :mask.size(1), :, :])
                        masks[name] = mask
                    elif isinstance(module, nl.SharableLinear):
                        mask = torch.ByteTensor(module.weight.data.size()).fill_(0)
                        if 'cuda' in module.weight.data.type():
                            mask = mask.cuda()
                        mask[:, :].copy_(masks[name][:mask.size(0), :mask.size(1)])
                        masks[name] = mask

    if args.dataset not in shared_layer_info:

        shared_layer_info[args.dataset] = {
            'bias': {},
            'bn_layer_running_mean': {},
            'bn_layer_running_var': {},
            'bn_layer_weight': {},
            'bn_layer_bias': {},
            'piggymask': {}
        }

        piggymasks = {}
        task_id = model.module.datasets.index(args.dataset) + 1
        if task_id > 1:
            for name, module in model.module.named_modules():
                if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                    piggymasks[name] = torch.zeros_like(masks['module.' + name], dtype=torch.float32)
                    piggymasks[name].fill_(0.01)
                    piggymasks[name] = Parameter(piggymasks[name])
                    module.piggymask = piggymasks[name]        
    else:
        piggymasks = shared_layer_info[args.dataset]['piggymask']
        task_id = model.module.datasets.index(args.dataset) + 1
        if task_id > 1:
            for name, module in model.module.named_modules():
                if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                    module.piggymask = piggymasks[name]

    shared_layer_info[args.dataset]['network_width_multiplier'] = args.network_width_multiplier

    if args.num_classes == 2:
        train_loader = dataset.cifar100_train_loader_two_class(args.dataset, args.batch_size)
        val_loader = dataset.cifar100_val_loader_two_class(args.dataset, args.val_batch_size)
    elif args.num_classes == 5:
        train_loader = dataset.cifar100_train_loader(args.dataset, args.batch_size)
        val_loader = dataset.cifar100_val_loader(args.dataset, args.val_batch_size)
    else:
        print("num_classes should be either 2 or 5")
        sys.exit(1)
        
    # if we are going to save checkpoint in other folder, then we recalculate the starting epoch
    if args.save_folder != args.load_folder:
        start_epoch = 0
    else:
        start_epoch = resume_from_epoch

    curr_prune_step = begin_prune_step = start_epoch * len(train_loader)
    end_prune_step = curr_prune_step + args.pruning_interval * len(train_loader)

    manager = Manager(args, model, shared_layer_info, masks, train_loader, val_loader, begin_prune_step, end_prune_step)
    if args.mode == 'inference':
        manager.load_checkpoint_only_for_evaluate(resume_from_epoch, resume_folder)
        manager.validate(resume_from_epoch-1)
        return

    lr = args.lr
    lr_mask = args.lr_mask
    # update all layers
    named_params = dict(model.named_parameters())
    params_to_optimize_via_SGD = []
    named_of_params_to_optimize_via_SGD = []
    masks_to_optimize_via_Adam = []
    named_of_masks_to_optimize_via_Adam = []

    for name, param in named_params.items():
        if 'classifiers' in name:
            if '.{}.'.format(model.module.datasets.index(args.dataset)) in name:
                params_to_optimize_via_SGD.append(param)
                named_of_params_to_optimize_via_SGD.append(name)                
            continue
        elif 'piggymask' in name:
            masks_to_optimize_via_Adam.append(param)
            named_of_masks_to_optimize_via_Adam.append(name)
        else:
            params_to_optimize_via_SGD.append(param)
            named_of_params_to_optimize_via_SGD.append(name)

    optimizer_network = optim.SGD(params_to_optimize_via_SGD, lr=lr,
                          weight_decay=0.0, momentum=0.9, nesterov=True)
    optimizers = Optimizers()
    optimizers.add(optimizer_network, lr)

    if masks_to_optimize_via_Adam:
        optimizer_mask = optim.Adam(masks_to_optimize_via_Adam, lr=lr_mask)
        optimizers.add(optimizer_mask, lr_mask)

    manager.load_checkpoint(optimizers, resume_from_epoch, resume_folder)

    # total_elements = 0
    # total_zeros_elements = 0
    # for name, module in model.named_modules():
    #     if isinstance(module, nl.SharableConv2d):
    #         zero_channels = module.piggymask.le(args.threshold).sum()
    #         zero_elements = module.weight.data.numel()/module.piggymask.size(0)*zero_channels
    #         total_zeros_elements += zero_elements
    #         total_elements += module.weight.data.numel()
    #         print('{}: channel level: num_zeros {}, total {}; '
    #                   'element level: num_zeros {}, total {}'.format(
    #                     name, zero_channels, module.piggymask.size(0),
    #                           zero_elements, module.weight.data.numel()))

    #         # zero_elements = module.piggymask.le(args.threshold).sum()
    #         # total_zeros_elements += zero_elements
    #         # total_elements += module.weight.data.numel()
    #         # print('{}: element level: num_zeros {}, total {}'.format(
    #         #             name, zero_elements, module.piggymask.numel()))
    # print('pruning ratio: {}'.format(float(total_zeros_elements)/total_elements))
    # pdb.set_trace()

    """Performs training."""
    curr_lrs = []
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            curr_lrs.append(param_group['lr'])
            break

    if start_epoch != 0:
        curr_best_accuracy = manager.validate(start_epoch-1)
    else:
        curr_best_accuracy = 0.0

    if args.jsonfile is None or not os.path.isfile(args.jsonfile):
        sys.exit(3)
    with open(args.jsonfile, 'r') as jsonfile: 
        json_data = json.load(jsonfile)
        baseline_acc = float(json_data[args.dataset])
    
    if args.mode == 'prune':
        # with open(os.path.join(os.getcwd(), args.tmp_benchmark_file), 'r') as jsonfile:
        #         json_data = json.load(jsonfile)
        #         acc_before_prune = float(json_data['acc_before_prune'])
        
        # if acc_before_prune - baseline_acc > 0.01:
        #     history_best_avg_val_acc_when_prune = acc_before_prune - 0.015
        # else:
        #     history_best_avg_val_acc_when_prune = acc_before_prune - 0.01
        history_best_avg_val_acc_when_prune = baseline_acc - 0.01

        stop_prune = True

        if 'gradual_prune' in args.load_folder and args.save_folder == args.load_folder:
            args.epochs = 20 + resume_from_epoch
        print()
        print('Before pruning: ')
        print('Sparsity range: {} -> {}'.format(args.initial_sparsity, args.target_sparsity))
        curr_best_accuracy = manager.validate(start_epoch-1)
        print()

    elif args.mode == 'finetune':
        manager.pruner.make_finetuning_mask()
        history_best_avg_val_acc = 0.0

    for epoch_idx in range(start_epoch, args.epochs):
        avg_train_acc, curr_prune_step = manager.train(optimizers, epoch_idx, curr_lrs, curr_prune_step)

        avg_val_acc = manager.validate(epoch_idx)

        if args.mode == 'prune' and (epoch_idx+1) >= (args.pruning_interval + start_epoch) and (
            avg_val_acc > history_best_avg_val_acc_when_prune):
            stop_prune = False
            history_best_avg_val_acc_when_prune = avg_val_acc
            if args.save_folder is not None:
                paths = os.listdir(args.save_folder)
                if paths and '.pth.tar' in paths[0]:
                    for checkpoint_file in paths:
                        os.remove(os.path.join(args.save_folder, checkpoint_file))
            else:
                print('Something is wrong! Block the program with pdb')
                pdb.set_trace()                

            manager.save_checkpoint(optimizers, epoch_idx, args.save_folder)

        if args.mode == 'finetune':        
            if epoch_idx + 1 == 50 or epoch_idx + 1 == 80:
                for param_group in optimizers[0].param_groups:
                    param_group['lr'] *= 0.1
                curr_lrs[0] = param_group['lr']
            if len(optimizers.lrs) == 2 and epoch_idx + 1 == 50:
                for param_group in optimizers[1].param_groups:
                    param_group['lr'] *= 0.2
                curr_lrs[1] = param_group['lr']

            if avg_val_acc > history_best_avg_val_acc:
                if args.save_folder is not None:
                    paths = os.listdir(args.save_folder)
                    if paths and '.pth.tar' in paths[0]:
                        for checkpoint_file in paths:
                            os.remove(os.path.join(args.save_folder, checkpoint_file))
                else:
                    print('Something is wrong! Block the program with pdb')
                    pdb.set_trace()                

                history_best_avg_val_acc = avg_val_acc
                manager.save_checkpoint(optimizers, epoch_idx, args.save_folder)

    print('-' * 16)

    if args.mode == 'finetune' and not args.test_piggymask:
        if avg_train_acc > 0.95 and (history_best_avg_val_acc - baseline_acc) > -0.01:
            # json_data = {}
            # json_data['acc_before_prune'] = '{:.4f}'.format(history_best_avg_val_acc)
            # with open(args.tmp_benchmark_file, 'w') as jsonfile:
            #     json.dump(json_data, jsonfile)
            pass
        else:
            print("It's time to expand the Network")
            print('Auto expand network')
            sys.exit(2)

    elif args.mode == 'prune' and stop_prune:
        print('Acc too low, stop pruning.')
        sys.exit(4)

if __name__ == '__main__':
  main()

