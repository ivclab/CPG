import UTILS.dataset
import torch
import torch.nn as nn
import torch.optim as optim
from UTILS.prune import SparsePruner
from tqdm import tqdm
import pdb
from pprint import pprint
import os
import math
from datetime import datetime
import models.layers as nl
# import imdbfolder_coco as imdbfolder
import sys
import logging

class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val, num):
        self.sum += val * num
        self.n += num

    @property
    def avg(self):
        return self.sum / self.n

def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()

class Manager(object):
    """Handles training and pruning."""

    def __init__(self, args, model, shared_layer_info, masks, train_loader, val_loader, begin_prune_step, end_prune_step):
        self.args = args
        self.model = model
        self.shared_layer_info = shared_layer_info
        self.inference_dataset_idx = self.model.module.datasets.index(args.dataset) + 1        
        self.pruner = SparsePruner(self.model, masks, self.args, begin_prune_step, end_prune_step, self.inference_dataset_idx)

        self.train_loader = train_loader
        self.val_loader = val_loader
        # Set up data loader, criterion, and pruner.
        # if 'cifar10' in args.train_path:
        #     self.train_loader = dataset.cifar10_train_loader('/home/ivclab/fevemania/prac_DL/shrink_and_expand/data', args.batch_size, pin_memory=args.cuda)
        #     self.val_loader = dataset.cifar10_val_loader('/home/ivclab/fevemania/prac_DL/shrink_and_expand/data', args.batch_size, pin_memory=args.cuda)
        # else:
        #     if 'cropped' in args.train_path:
        #         train_loader = dataset.train_loader_cropped
        #         val_loader = dataset.val_loader_cropped
        #     else:
        #         train_loader = dataset.train_loader
        #         val_loader = dataset.val_loader
            
        #     self.train_loader = train_loader(
        #         args.train_path, args.batch_size, num_workers=args.workers, pin_memory=args.cuda)
        #     self.val_loader = val_loader(
        #         args.val_path, args.val_batch_size, num_workers=args.workers, pin_memory=args.cuda)

        # train_loaders, val_loaders, num_classes = imdbfolder.prepare_data_loaders(dataset_names=[args.dataset], 
        #                                         data_dir=args.datadir, imdb_dir=args.imdbdir, shuffle_train=True, 
        #                                         pin_memory=args.cuda, batch_size=args.batch_size,
        #                                         val_batch_size=args.val_batch_size)
        # args.num_classes = num_classes[0]
        # self.train_loader = train_loaders[0]
        # self.val_loader = val_loaders[0]

        self.criterion = nn.CrossEntropyLoss()

    def train(self, optimizers, epoch_idx, curr_lrs, curr_prune_step):
        # Set model to training mode
        self.model.train()

        train_loss = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')
        
        with tqdm(total=len(self.train_loader),
                  desc='Train Ep. #{}: '.format(epoch_idx + 1), 
                        #datetime.strftime(datetime.now(), '%Y/%m/%d-%H:%M:%S')),
                  disable=False,
                  ascii=True) as t:
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()

                optimizers.zero_grad()
                # Do forward-backward.
                output = self.model(data)
                num = len(data)
                train_accuracy.update(accuracy(output, target), num)
                loss = self.criterion(output, target)
                train_loss.update(loss, num)
                loss.backward()

                # Set fixed param grads to 0.
                self.pruner.do_weight_decay_and_make_grads_zero()
                # Gradient is applied across all ranks
                optimizers.step()
      
                # Set pruned weights to 0.
                if self.args.mode == 'prune':
                    self.pruner.gradually_prune(curr_prune_step)
                    curr_prune_step += 1

                if self.inference_dataset_idx == 1:
                    t.set_postfix({'loss': train_loss.avg.item(),
                                   'accuracy': '{:.2f}'.format(100. * train_accuracy.avg.item()),
                                   'lr': curr_lrs[0],
                                   'sparsity': self.pruner.calculate_sparsity(),
                                   'network_width_mpl': self.args.network_width_multiplier})
                else:
                    t.set_postfix({'loss': train_loss.avg.item(),
                                   'accuracy': '{:.2f}'.format(100. * train_accuracy.avg.item()),
                                   'lr': curr_lrs[0],
                                   # 'lr_mask': curr_lrs[1],
                                   'sparsity': self.pruner.calculate_sparsity(),
                                   'network_width_mpl': self.args.network_width_multiplier})                    
                t.update(1)
        
        if self.args.log_path:
            summary = {'loss': '{:.3f}'.format(train_loss.avg.item()), 
                       'accuracy': '{:.2f}'.format(100 * train_accuracy.avg.item()),
                       'lr': curr_lrs[0],
                       'sparsity': '{:.3f}'.format(self.pruner.calculate_sparsity()),
                       'network_width_mpl': self.args.network_width_multiplier}
            logging.info('In train()-> Train Ep. #{} '.format(epoch_idx + 1) + ', '.join(['{}: {}'.format(k, v) for k, v in summary.items()]))

        return train_accuracy.avg.item(), curr_prune_step

    def validate(self, epoch_idx, biases=None):
        """Performs evaluation."""
        self.pruner.apply_mask()
        self.model.eval()
        val_loss = Metric('val_loss')
        val_accuracy = Metric('val_accuracy')

        with tqdm(total=len(self.val_loader),
                  desc='Val Ep. #{}: '.format(epoch_idx + 1), #, datetime.strftime(datetime.now(), '%Y/%m/%d-%H:%M:%S'))
                  ascii=True) as t:
            with torch.no_grad():
                for data, target in self.val_loader:
                    if self.args.cuda:
                        data, target = data.cuda(), target.cuda()

                    output = self.model(data)
                    num = len(data)
                    val_loss.update(self.criterion(output, target), num)
                    val_accuracy.update(accuracy(output, target), num)

                    if self.inference_dataset_idx == 1:
                        t.set_postfix({'loss': val_loss.avg.item(),
                                       'accuracy': '{:.2f}'.format(100. * val_accuracy.avg.item()),
                                       'sparsity': self.pruner.calculate_sparsity(),
                                       'task{} ratio'.format(self.inference_dataset_idx): self.pruner.calculate_curr_task_ratio(),
                                       'zero ratio': self.pruner.calculate_zero_ratio(),
                                       'mpl': self.args.network_width_multiplier})
                    else:
                        t.set_postfix({'loss': val_loss.avg.item(),
                                       'accuracy': '{:.2f}'.format(100. * val_accuracy.avg.item()),
                                       'sparsity': self.pruner.calculate_sparsity(),
                                       'task{} ratio'.format(self.inference_dataset_idx): self.pruner.calculate_curr_task_ratio(),
                                       'shared_ratio': self.pruner.calculate_shared_part_ratio(),
                                       'zero ratio': self.pruner.calculate_zero_ratio(),
                                       'mpl': self.args.network_width_multiplier})
                    t.update(1)
        
        if self.args.log_path:
            if self.inference_dataset_idx != 1:
                summary = {'loss': '{:.3f}'.format(val_loss.avg.item()), 
                           'accuracy': '{:.2f}'.format(100 * val_accuracy.avg.item()),
                           'sparsity': '{:.3f}'.format(self.pruner.calculate_sparsity()),
                           'task{} ratio'.format(self.inference_dataset_idx): '{:.3f}'.format(self.pruner.calculate_curr_task_ratio()),
                           'zero ratio': '{:.4f}'.format(self.pruner.calculate_zero_ratio()),
                           'shared_ratio': '{:.6f}'.format(self.pruner.calculate_shared_part_ratio()),
                           'mpl': '{:.3f}'.format(self.args.network_width_multiplier)}
            else:
                summary = {'loss': '{:.3f}'.format(val_loss.avg.item()), 
                           'accuracy': '{:.2f}'.format(100 * val_accuracy.avg.item()),
                           'sparsity': '{:.3f}'.format(self.pruner.calculate_sparsity()),
                           'task{} ratio'.format(self.inference_dataset_idx): '{:.3f}'.format(self.pruner.calculate_curr_task_ratio()),
                           'zero ratio': '{:.4f}'.format(self.pruner.calculate_zero_ratio()),
                           'mpl': '{:.3f}'.format(self.args.network_width_multiplier)}
               
            logging.info('In validate()-> Val Ep. #{} '.format(epoch_idx + 1) + ', '.join(['{}: {}'.format(k, v) for k, v in summary.items()]))

        return val_accuracy.avg.item()

    def save_checkpoint(self, optimizers, epoch_idx, save_folder):
        """Saves model to file."""
        filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=epoch_idx + 1)

        self.shared_layer_info[self.args.dataset]['bias'] = {}
        for name, module in self.model.module.named_modules():
            if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                if hasattr(module, 'bias'):
                    self.shared_layer_info[self.args.dataset][
                        'bias'][name] = module.bias
                elif hasattr(module, 'conv_bias'):
                    self.share_layer_info[self.args.dataset]['bias'][name] = module.conv_bias
    
                elif hasattr(module, 'fc_bias'):
                    self.share_layer_info[self.args.dataset]['bias'][name] = module.fc_bias

                if module.piggymask is not None:
                    self.shared_layer_info[self.args.dataset][
                        'piggymask'][name] = module.piggymask

            elif isinstance(module, nn.BatchNorm2d):
                self.shared_layer_info[self.args.dataset][
                    'bn_layer_running_mean'][name] = module.running_mean
                self.shared_layer_info[self.args.dataset][
                    'bn_layer_running_var'][name] = module.running_var
                self.shared_layer_info[self.args.dataset][
                    'bn_layer_weight'][name] = module.weight
                self.shared_layer_info[self.args.dataset][
                    'bn_layer_bias'][name] = module.bias

        checkpoint = {
            'model_state_dict': self.model.module.state_dict(),
            'dataset_history': self.model.module.datasets,
            'dataset2num_classes': self.model.module.dataset2num_classes,
            'masks': self.pruner.masks,
            'shared_layer_info': self.shared_layer_info
            # 'optimizer_network_state_dict': optimizers[0].state_dict(),
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, optimizers, resume_from_epoch, save_folder, restore_optimizers=True):
        # Restore from a previous checkpoint, if initial_epoch is specified.
        # Horovod: restore on the first worker which will broadcast weights to other workers.
        if resume_from_epoch > 0:
            filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=resume_from_epoch)
            checkpoint = torch.load(filepath)
            checkpoint_keys = checkpoint.keys()
            state_dict = checkpoint['model_state_dict']
            curr_model_state_dict = self.model.module.state_dict()
            for name, param in state_dict.items():
                if 'piggymask' in name or name == 'classifier.weight' or name == 'classifier.bias':
                    continue
                elif len(curr_model_state_dict[name].size()) == 4:
                    # Conv layer
                    curr_model_state_dict[name][:param.size(0), :param.size(1), :, :].copy_(param)
                elif len(curr_model_state_dict[name].size()) == 2 and 'features' in name:
                    # FC conv (feature layer)
                    curr_model_state_dict[name][:param.size(0), :param.size(1)].copy_(param)
                elif len(curr_model_state_dict[name].size()) == 1:
                    # bn layer
                    curr_model_state_dict[name][:param.size(0)].copy_(param)
                elif 'classifiers' in name:  ################
                    #pdb.set_trace()
                    curr_model_state_dict[name][:param.size(0), :param.size(1)].copy_(param)
                else:
                    try:
                        curr_model_state_dict[name].copy_(param)
                    except:
                        pdb.set_trace()
                        print("There is some corner case that we haven't tackled")

            # if restore_optimizers:
            #     if 'optimizer_network_state_dict' in checkpoint_keys:
            #         optimizers[0].load_state_dict(checkpoint['optimizer_network_state_dict'])
                    # optimizers[1].load_state_dict(checkpoint['optimizer_mask_state_dict'])
            
    def load_checkpoint_only_for_evaluate(self, resume_from_epoch, save_folder):
        if resume_from_epoch > 0:
            filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=resume_from_epoch)
            checkpoint = torch.load(filepath)
            checkpoint_keys = checkpoint.keys()
            state_dict = checkpoint['model_state_dict']

            curr_model_state_dict = self.model.module.state_dict()

            for name, param in state_dict.items():
                if 'piggymask' in name: # we load piggymask value in main.py
                    continue                                                
                if name == 'classifier.weight' or name == 'classifier.bias':
                    continue
                elif len(curr_model_state_dict[name].size()) == 4:
                    # Conv layer
                    curr_model_state_dict[name].copy_(param[:curr_model_state_dict[name].size(0), :curr_model_state_dict[name].size(1), :, :])
                elif len(curr_model_state_dict[name].size()) == 2 and 'features' in name:
                    # FC conv (feature layer)
                    curr_model_state_dict[name].copy_(param[:curr_model_state_dict[name].size(0), :curr_model_state_dict[name].size(1)])
                elif len(curr_model_state_dict[name].size()) == 1:
                    # bn layer
                    curr_model_state_dict[name].copy_(param[:curr_model_state_dict[name].size(0)])
                else:
                    curr_model_state_dict[name].copy_(param)


            # load the batch norm params and bias in convolution in correspond to curr dataset
            for name, module in self.model.module.named_modules():
                if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                    if module.bias is not None:
                        module.bias = self.shared_layer_info[self.args.dataset]['bias'][name]

                elif isinstance(module, nn.BatchNorm2d):
                    module.running_mean = self.shared_layer_info[self.args.dataset][
                        'bn_layer_running_mean'][name]
                    module.running_var = self.shared_layer_info[self.args.dataset][
                        'bn_layer_running_var'][name]
                    module.weight = self.shared_layer_info[self.args.dataset][
                        'bn_layer_weight'][name]
                    module.bias = self.shared_layer_info[self.args.dataset][
                        'bn_layer_bias'][name]
