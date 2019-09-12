"""Handles all the pruning-related stuff."""
from __future__ import print_function

import collections

import numpy as np

import torch
import torch.nn as nn
import pdb
import models_with_one_mask.layers as nl
import sys

class SparsePruner(object):
    """Performs pruning on the given model."""

    def __init__(self, model, masks, args, begin_prune_step, end_prune_step, inference_dataset_idx):
        self.model = model
        self.args = args
        self.sparsity_func_exponent = 3
        self.begin_prune_step = begin_prune_step
        self.end_prune_step = end_prune_step
        self.last_prune_step = begin_prune_step

        self.masks = masks
        #valid_key = list(masks.keys())[0]
        #self.current_dataset_idx = max(len(self.model.module.datasets) - 1, self.masks[valid_key].max()) #TODO
        if args.mode == 'prune' or args.mode == 'inference' or (args.mode == 'finetune' and args.finetune_again):
            self.current_dataset_idx = self.model.module.datasets.index(args.dataset) + 1
        elif args.mode == 'finetune':
            self.current_dataset_idx = len(self.model.module.datasets) - 1
        else:
            print('Dont support mode: \'{}\''.format(args.mode))
            sys.exit(-1)

        self.inference_dataset_idx = inference_dataset_idx

    def _pruning_mask(self, weights, mask, layer_name, pruning_ratio):
        """Ranks weights by magnitude. Sets all below kth to 0.
           Returns pruned mask.
        """
        # Select all prunable weights, ie. belonging to current dataset.
        tensor = weights[mask.eq(self.current_dataset_idx) | mask.eq(0)] # This will flatten weights
        abs_tensor = tensor.abs()
        cutoff_rank = round(pruning_ratio * tensor.numel())
        try:
            cutoff_value = abs_tensor.cpu().kthvalue(cutoff_rank)[0].cuda() # value at cutoff rank
        except:
            print("Not enough weights for pruning, that is to say, too little space for new task, need expand the network.")
            sys.exit(2)
        # Remove those weights which are below cutoff and belong to current
        # dataset that we are training for.
        remove_mask = weights.abs().le(cutoff_value) * mask.eq(self.current_dataset_idx)

        # mask = 1 - remove_mask
        mask[remove_mask.eq(1)] = 0
        # print('Layer {}, pruned {}/{} ({:.2f}%)'.format(
        #        layer_name, mask.eq(0).sum(), tensor.numel(),
        #        float(100 * mask.eq(0).sum()) / tensor.numel()))

        return mask

    def _adjust_sparsity(self, curr_prune_step):

        p = min(1.0,
                max(0.0, 
                    ((curr_prune_step - self.begin_prune_step) 
                    / (self.end_prune_step - self.begin_prune_step)) 
                ))

        sparsity = self.args.target_sparsity + \
            (self.args.initial_sparsity - self.args.target_sparsity) * pow(1-p, self.sparsity_func_exponent)

        return sparsity

    def _time_to_update_masks(self, curr_prune_step):
        is_step_within_pruning_range = \
            (curr_prune_step >= self.begin_prune_step) and \
            (curr_prune_step <= self.end_prune_step)

        is_pruning_step = (
            self.last_prune_step + self.args.pruning_frequency) <= curr_prune_step

        return is_step_within_pruning_range and is_pruning_step        

    def gradually_prune(self, curr_prune_step):
        if self._time_to_update_masks(curr_prune_step):
            self.last_prune_step = curr_prune_step    
            curr_pruning_ratio = self._adjust_sparsity(curr_prune_step)
            # print('Pruning for dataset idx: %d' % (self.current_dataset_idx))
            # print('Pruning each layer by removing {:.2f}% of values'.format(100 * curr_pruning_ratio))

            for name, module in self.model.module.named_modules():
                if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                    mask = self._pruning_mask(module.weight.data, self.masks[name], name, pruning_ratio=curr_pruning_ratio)
                    self.masks[name] = mask
                    module.weight.data[self.masks[name].eq(0)] = 0.0
        else:
            curr_pruning_ratio = self._adjust_sparsity(self.last_prune_step)


        return curr_pruning_ratio

    def one_shot_prune(self, one_shot_prune_perc):
        """Gets pruning mask for each layer, based on previous_masks.
           Sets the self.current_masks to the computed pruning masks.
        """
        print('Pruning for dataset idx: %d' % (self.current_dataset_idx))
        print('Pruning each layer by removing %.2f%% of values' % (100 * one_shot_prune_perc))

        for name, module in self.model.module.named_modules():
            if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                mask = self._pruning_mask(
                    module.weight.data, self.masks[name], name, pruning_ratio=one_shot_prune_perc)
                self.masks[name] = mask

                # Set pruned weights to 0.
                module.weight.data[self.masks[name].eq(0)] = 0.0


    def calculate_sparsity(self):
        total_elem = 0
        zero_elem = 0
        is_first_conv = True 

        for name, module in self.model.module.named_modules():
            if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                if is_first_conv:
                    is_first_conv = False
                    continue
                mask = self.masks[name]
                total_elem += torch.sum(mask.eq(self.inference_dataset_idx) | mask.eq(0))
                zero_elem += torch.sum(mask.eq(0))

        if total_elem.cpu() != 0.0:
            return float(zero_elem.cpu()) / float(total_elem.cpu())
        else:
            return 0.0

    def calculate_curr_task_ratio(self):
        total_elem = 0
        curr_task_elem = 0

        for name, module in self.model.module.named_modules():
            if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):   
                mask = self.masks[name]
                total_elem += mask.numel()
                curr_task_elem += torch.sum(mask.eq(self.inference_dataset_idx))

        return float(curr_task_elem.cpu()) / total_elem #* self.args.network_width_multiplier

    def calculate_zero_ratio(self):
        total_elem = 0
        zero_elem = 0
 
        for name, module in self.model.module.named_modules():
            if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                mask = self.masks[name]
                total_elem += mask.numel()
                zero_elem += torch.sum(mask.eq(0))

        return float(zero_elem.cpu()) / total_elem # * self.args.network_width_multiplier

    def calculate_shared_part_ratio(self):
        total_elem = 0
        shared_elem = 0
        for name, module in self.model.module.named_modules():
            if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                mask = self.masks[name]
                total_elem += torch.sum((mask < self.current_dataset_idx) & (0 < mask))
                shared_elem += torch.sum((mask < self.current_dataset_idx) & (0 < mask) & (module.piggymask_task_tag == self.current_dataset_idx) & (module.piggymask_float > 5e-3))

        if total_elem.cpu() != 0.0:
            return float(shared_elem.cpu()) / float(total_elem.cpu()) #* (self.model.module.shared_layer_info[self.model.module.datasets[-2]]['network_width_multiplier'] ** 2)
        else:
            return 0.0

    def do_weight_decay_and_make_grads_zero(self):
        """Sets grads of fixed weights to 0."""
        assert self.masks
        for name, module in self.model.module.named_modules():
            if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                mask = self.masks[name]
                # Set grads of all weights not belonging to current dataset to 0.
                if module.weight.grad is not None:
                    module.weight.grad.data.add_(self.args.weight_decay, module.weight.data)
                    module.weight.grad.data[mask.ne(
                        self.current_dataset_idx)] = 0
                if module.piggymask_float is not None and module.piggymask_float.grad is not None:
                    if self.args.mode == 'finetune':
                        module.piggymask_float.grad.data[module.piggymask_task_tag.data.ne(self.current_dataset_idx)] = 0
                    elif self.args.mode == 'prune':
                        module.piggymask_float.grad.data.fill_(0)

    def make_pruned_zero(self):
        """Makes pruned weights 0."""
        assert self.masks

        for name, module in self.model.module.named_modules():
            if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                layer_mask = self.masks[name]
                module.weight.data[layer_mask.eq(0)] = 0.0

    def apply_mask(self):
        """To be done to retrieve weights just for a particular dataset."""
        for name, module in self.model.module.named_modules():
            if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                weight = module.weight.data
                mask = self.masks[name].cuda()
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(self.inference_dataset_idx)] = 0.0

    def make_finetuning_mask(self):
        """Turns previously pruned weights into trainable weights for
           current dataset.
        """
        assert self.masks
        self.current_dataset_idx += 1

        for name, module in self.model.module.named_modules():
            if isinstance(module, nl.SharableConv2d) or isinstance(module, nl.SharableLinear):
                mask = self.masks[name]
                reinitial_tensor = torch.zeros_like(module.weight)
                nn.init.kaiming_normal_(reinitial_tensor, mode='fan_out', nonlinearity='relu')
                module.weight.data[mask.eq(0) & module.weight.data.eq(0.0)] = reinitial_tensor.data[mask.eq(0) & module.weight.data.eq(0.0)]
                mask[mask.eq(0)] = self.current_dataset_idx
