import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .prune import SparsePruner
from tqdm import tqdm
import pdb
from pprint import pprint
import os
import math
from datetime import datetime
import models.layers as nl
# import imdbfolder_coco as imdbfolder
import sys
from torch.autograd import Variable
from .metrics import fv_evaluate
from models import AngleLoss

class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += val
        self.n += 1

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
        self.val_loader   = val_loader

        if args.dataset == 'face_verification':
            self.criterion = AngleLoss()
        elif args.dataset == 'emotion':
            class_counts = torch.from_numpy(np.array([74874, 134415, 25459, 14090, 6378, 3803, 24882]).astype(np.float32))
            class_weights = (torch.sum(class_counts) - class_counts) / class_counts
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.cuda())
        else:
            self.criterion = nn.CrossEntropyLoss()

    def train(self, optimizers, epoch_idx, curr_lrs, curr_prune_step):
        # Set model to training mode
        self.model.train()

        train_loss     = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')

        with tqdm(total=len(self.train_loader),
                  desc='Train Ep. #{}: '.format(epoch_idx + 1),
                  disable=False,
                  ascii=True) as t:
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()

                optimizers.zero_grad()
                # Do forward-backward.
                output = self.model(data)

                if self.args.dataset != 'face_verification':
                    train_accuracy.update(accuracy(output, target))

                loss = self.criterion(output, target)
                train_loss.update(loss)
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
                                   'sparsity': self.pruner.calculate_sparsity(),
                                   'network_width_mpl': self.args.network_width_multiplier})
                t.update(1)

        summary = {'loss': '{:.3f}'.format(train_loss.avg.item()),
                   'accuracy': '{:.2f}'.format(100. * train_accuracy.avg.item()),
                   'lr': curr_lrs[0],
                   'sparsity': '{:.3f}'.format(self.pruner.calculate_sparsity()),
                   'network_width_mpl': self.args.network_width_multiplier}
        logging.info(('In train()-> Train Ep. #{} '.format(epoch_idx + 1)
                     + ', '.join(['{}: {}'.format(k, v) for k, v in summary.items()])))
        return train_accuracy.avg.item(), curr_prune_step

    #{{{ Evaluate classification
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

                    val_loss.update(self.criterion(output, target))
                    val_accuracy.update(accuracy(output, target))

                    if self.inference_dataset_idx == 1:
                        t.set_postfix({'loss': val_loss.avg.item(),
                                       'accuracy': '{:.2f}'.format(100. * val_accuracy.avg.item()),
                                       'sparsity': self.pruner.calculate_sparsity(),
                                       'task{} ratio'.format(self.inference_dataset_idx): '{:.2f}'.format(self.pruner.calculate_curr_task_ratio()),
                                       'zero ratio': self.pruner.calculate_zero_ratio()})
                    else:
                        t.set_postfix({'loss': val_loss.avg.item(),
                                       'accuracy': '{:.2f}'.format(100. * val_accuracy.avg.item()),
                                       'sparsity': self.pruner.calculate_sparsity(),
                                       'task{} ratio'.format(self.inference_dataset_idx): '{:.2f}'.format(self.pruner.calculate_curr_task_ratio()),
                                       'shared_ratio': self.pruner.calculate_shared_part_ratio(),
                                       'zero ratio': self.pruner.calculate_zero_ratio()})
                    t.update(1)
        summary = {'loss': '{:.3f}'.format(val_loss.avg.item()),
                   'accuracy': '{:.2f}'.format(100. * val_accuracy.avg.item()),
                   'sparsity': '{:.3f}'.format(self.pruner.calculate_sparsity()),
                   'task{} ratio'.format(self.inference_dataset_idx): '{:.3f}'.format(self.pruner.calculate_curr_task_ratio()),
                   'zero ratio': '{:.3f}'.format(self.pruner.calculate_zero_ratio())}
        if self.inference_dataset_idx != 1:
            summary['shared_ratio'] = '{:.3f}'.format(self.pruner.calculate_shared_part_ratio())
        logging.info(('In validate()-> Val Ep. #{} '.format(epoch_idx + 1)
                     + ', '.join(['{}: {}'.format(k, v) for k, v in summary.items()])))
        return val_accuracy.avg.item()
    #}}}

    #{{{ Evaluate LFW
    def evalLFW(self, epoch_idx):
        distance_metric = True
        subtract_mean   = False
        self.pruner.apply_mask()
        self.model.eval() # switch to evaluate mode
        labels, embedding_list_a, embedding_list_b = [], [], []
        with torch.no_grad():
            with tqdm(total=len(self.val_loader),
                      desc='Validate Epoch  #{}: '.format(epoch_idx + 1),
                      ascii=True) as t:
                for batch_idx, (data_a, data_p, label) in enumerate(self.val_loader):
                    data_a, data_p = data_a.cuda(), data_p.cuda()
                    data_a, data_p, label = Variable(data_a, volatile=True), \
                                            Variable(data_p, volatile=True), Variable(label)
                    # ==== compute output ====
                    out_a = self.model.module.forward_to_embeddings(data_a)
                    out_p = self.model.module.forward_to_embeddings(data_p)
                    # do L2 normalization for features
                    if not distance_metric:
                        out_a = F.normalize(out_a, p=2, dim=1)
                        out_p = F.normalize(out_p, p=2, dim=1)
                    out_a = out_a.data.cpu().numpy()
                    out_p = out_p.data.cpu().numpy()

                    embedding_list_a.append(out_a)
                    embedding_list_b.append(out_p)
                    # ========================
                    labels.append(label.data.cpu().numpy())
                    t.update(1)

        labels = np.array([sublabel for label in labels for sublabel in label])
        embedding_list_a = np.array([item for embedding in embedding_list_a for item in embedding])
        embedding_list_b = np.array([item for embedding in embedding_list_b for item in embedding])
        tpr, fpr, accuracy, val, val_std, far = fv_evaluate(embedding_list_a, embedding_list_b, labels,
                                                distance_metric=distance_metric, subtract_mean=subtract_mean)
        print('In evalLFW(): Test set: Accuracy: {:.5f}+-{:.5f}'.format(np.mean(accuracy),np.std(accuracy)))
        logging.info(('In evalLFW()-> Validate Epoch #{} '.format(epoch_idx + 1)
                     + 'Test set: Accuracy: {:.5f}+-{:.5f}, '.format(np.mean(accuracy),np.std(accuracy))
                     + 'task_ratio: {:.2f}'.format(self.pruner.calculate_curr_task_ratio())))
        return np.mean(accuracy)
    #}}}

    def save_checkpoint(self, optimizers, epoch_idx, save_folder):
        """Saves model to file."""
        filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=epoch_idx + 1)

        for name, module in self.model.module.named_modules():
            if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                if module.bias is not None:
                    self.shared_layer_info[self.args.dataset][
                        'bias'][name] = module.bias
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
            elif isinstance(module, nn.PReLU):
                self.shared_layer_info[self.args.dataset][
                    'prelu_layer_weight'][name] = module.weight

        checkpoint = {
            'model_state_dict': self.model.module.state_dict(),
            'dataset_history': self.model.module.datasets,
            'dataset2num_classes': self.model.module.dataset2num_classes,
            'masks': self.pruner.masks,
            'shared_layer_info': self.shared_layer_info
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, optimizers, resume_from_epoch, save_folder, restore_optimizers=True):
        if resume_from_epoch > 0:
            filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=resume_from_epoch)
            checkpoint = torch.load(filepath)
            checkpoint_keys = checkpoint.keys()
            state_dict = checkpoint['model_state_dict']
            curr_model_state_dict = self.model.module.state_dict()
            for name, param in state_dict.items():
                if 'piggymask' in name or name == 'classifier.weight' or name == 'classifier.bias' or (name == 'classifier.0.weight' or name == 'classifier.0.bias' or name == 'classifier.1.weight'):
                    # I DONT WANT TO DO THIS! QQ That last 3 exprs are for anglelinear and embeddings
                    continue
                elif len(curr_model_state_dict[name].size()) == 4:
                    # Conv layer
                    curr_model_state_dict[name][:param.size(0), :param.size(1), :, :].copy_(param)
                elif len(curr_model_state_dict[name].size()) == 2 and 'features' in name:
                    # FC conv (feature layer)
                    curr_model_state_dict[name][:param.size(0), :param.size(1)].copy_(param)
                elif len(curr_model_state_dict[name].size()) == 1:
                    # bn and prelu layer
                    curr_model_state_dict[name][:param.size(0)].copy_(param)
                else:
                    curr_model_state_dict[name].copy_(param)

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
                if name == 'classifier.weight' or name == 'classifier.bias' or (name == 'classifier.0.weight' or name == 'classifier.0.bias' or name == 'classifier.1.weight'):
                     # I DONT WANT TO DO THIS! QQ That last 3 exprs are for anglelinear and embeddings
                    continue
                elif len(curr_model_state_dict[name].size()) == 4:
                    # Conv layer
                    curr_model_state_dict[name].copy_(param[:curr_model_state_dict[name].size(0), :curr_model_state_dict[name].size(1), :, :])
                elif len(curr_model_state_dict[name].size()) == 2 and 'features' in name:
                    # FC conv (feature layer)
                    curr_model_state_dict[name].copy_(param[:curr_model_state_dict[name].size(0), :curr_model_state_dict[name].size(1)])
                elif len(curr_model_state_dict[name].size()) == 1:
                    # bn and prelu layer
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
                elif isinstance(module, nn.PReLU):
                    module.weight = self.shared_layer_info[self.args.dataset][
                        'prelu_layer_weight'][name]
