import UTILS.dataset
import torch
import torch.nn as nn
import torch.optim as optim
from UTILS.packnet_prune import SparsePruner
from tqdm import tqdm
import pdb
from pprint import pprint
import os
import math
from datetime import datetime
# import imdbfolder_coco as imdbfolder

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

    def __init__(self, args, model, shared_layer_info, masks, train_loader, val_loader):
        self.args = args
        self.model = model
        self.shared_layer_info = shared_layer_info
        self.inference_dataset_idx = self.model.module.datasets.index(args.dataset) + 1
        self.pruner = SparsePruner(self.model, masks, self.args, None, None, self.inference_dataset_idx)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # train_loaders, val_loaders, num_classes = imdbfolder.prepare_data_loaders(dataset_names=[args.dataset], 
        #                                         data_dir=args.datadir, imdb_dir=args.imdbdir, shuffle_train=True, 
        #                                         pin_memory=args.cuda, batch_size=args.batch_size,
        #                                         val_batch_size=args.val_batch_size)
        # args.num_classes = num_classes[0]
        # self.train_loader = train_loaders[0]
        # self.val_loader = val_loaders[0]

        self.criterion = nn.CrossEntropyLoss()

    def train(self, optimizers, epoch_idx, curr_lrs):
        # Set model to training mode
        self.model.train()

        train_loss = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')
        
        with tqdm(total=len(self.train_loader),
                  desc='Train Epoch #{}: '.format(epoch_idx + 1), 
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
                self.pruner.make_pruned_zero()

                t.set_postfix({'loss': train_loss.avg.item(),
                               'accuracy': '{:.2f}'.format(100. * train_accuracy.avg.item()),
                               'lr': curr_lrs[0],
                               'sparsity': self.pruner.calculate_sparsity()})

                t.update(1)

        return train_accuracy.avg.item()

    def validate(self, epoch_idx, biases=None):
        """Performs evaluation."""
        self.pruner.apply_mask()
        self.model.eval()
        val_loss = Metric('val_loss')
        val_accuracy = Metric('val_accuracy')

        with tqdm(total=len(self.val_loader),
                  desc='Validate Epoch  #{}: '.format(epoch_idx + 1), #, datetime.strftime(datetime.now(), '%Y/%m/%d-%H:%M:%S'))
                  ascii=True) as t:
            with torch.no_grad():
                for data, target in self.val_loader:
                    if self.args.cuda:
                        data, target = data.cuda(), target.cuda()

                    output = self.model(data)
                    num = len(data)
                    val_loss.update(self.criterion(output, target), num)
                    val_accuracy.update(accuracy(output, target), num)

                    t.set_postfix({'loss': val_loss.avg.item(),
                                   'accuracy': '{:.2f}'.format(100. * val_accuracy.avg.item()),
                                   'sparsity': self.pruner.calculate_sparsity(),
                                   'task{} ratio'.format(self.inference_dataset_idx): self.pruner.calculate_curr_task_ratio(),
                                   'zero ratio': self.pruner.calculate_zero_ratio()})
                    t.update(1)

        return val_accuracy.avg.item()

    def one_shot_prune(self, one_shot_prune_perc):
        self.pruner.one_shot_prune(one_shot_prune_perc)

    def save_checkpoint(self, optimizers, epoch_idx, save_folder):
        """Saves model to file."""
        filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=epoch_idx + 1)

        for name, module in self.model.module.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.bias is not None:
                    self.shared_layer_info[self.args.dataset][
                        'conv_bias'][name] = module.bias
            elif isinstance(module, nn.BatchNorm2d):
                self.shared_layer_info[self.args.dataset][
                    'bn_layer_running_mean'][name] = module.running_mean
                self.shared_layer_info[self.args.dataset][
                    'bn_layer_running_var'][name] = module.running_var
                self.shared_layer_info[self.args.dataset][
                    'bn_layer_weight'][name] = module.weight
                self.shared_layer_info[self.args.dataset][
                    'bn_layer_bias'][name] = module.bias
            elif isinstance(module, nn.Linear) and 'features' in name:
                self.shared_layer_info[self.args.dataset]['fc_bias'][name] = module.bias

        checkpoint = {
            'model_state_dict': self.model.module.state_dict(),
            'dataset_history': self.model.module.datasets,
            'dataset2num_classes': self.model.module.dataset2num_classes,
            'masks': self.pruner.masks,
            'shared_layer_info': self.shared_layer_info,
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
                if name == 'classifier.weight' or name == 'classifier.bias':
                    continue
                else:
                    curr_model_state_dict[name].copy_(param)

            # if restore_optimizers:
            #     if 'optimizer_network_state_dict' in checkpoint_keys:
                    # optimizers[0].load_state_dict(checkpoint['optimizer_network_state_dict'])
            
    def load_checkpoint_for_inference(self, resume_from_epoch, save_folder):
        if resume_from_epoch > 0:
            filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=resume_from_epoch)
            checkpoint = torch.load(filepath)
            checkpoint_keys = checkpoint.keys()
            state_dict = checkpoint['model_state_dict']
            curr_model_state_dict = self.model.module.state_dict()
            for name, param in state_dict.items():
                if name == 'classifier.weight' or name == 'classifier.bias':
                    continue
                else:
                    curr_model_state_dict[name].copy_(param)

            # load the batch norm params and bias in convolution in correspond to curr dataset
            for name, module in self.model.module.named_modules():
                if isinstance(module, nn.Conv2d):
                    if module.bias is not None:
                        module.bias = self.shared_layer_info[self.args.dataset]['conv_bias'][name]
                elif isinstance(module, nn.BatchNorm2d):
                    module.running_mean = self.shared_layer_info[self.args.dataset][
                        'bn_layer_running_mean'][name]
                    module.running_var = self.shared_layer_info[self.args.dataset][
                        'bn_layer_running_var'][name]
                    module.weight = self.shared_layer_info[self.args.dataset][
                        'bn_layer_weight'][name]
                    module.bias = self.shared_layer_info[self.args.dataset][
                        'bn_layer_bias'][name]

                elif isinstance(module, nn.Linear) and 'features' in name:
                    module.bias = self.shared_layer_info[self.args.dataset]['fc_bias'][name]
