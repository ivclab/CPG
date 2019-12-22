import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from . import Metric, classification_accuracy
from .packnet_prune import SparsePruner
from .metrics import fv_evaluate
from packnet_models import AngleLoss


class Manager(object):
    """Handles training and pruning."""

    def __init__(self, args, model, shared_layer_info, masks, train_loader, val_loader):
        self.args  = args
        self.model = model
        self.shared_layer_info = shared_layer_info
        self.inference_dataset_idx = self.model.module.datasets.index(args.dataset) + 1
        self.pruner = SparsePruner(self.model, masks, self.args, None, None, self.inference_dataset_idx)
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
        return

    def train(self, optimizers, epoch_idx, curr_lrs):
        # Set model to training mode
        self.model.train()

        train_loss     = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')

        with tqdm(total=len(self.train_loader),
                  desc='Train Epoch #{}: '.format(epoch_idx + 1),
                  disable=False,
                  ascii=True) as t:
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if self.args.cuda:
                    data, target = data.cuda(), target.cuda()

                optimizers.zero_grad()
                # Do forward-backward.
                output = self.model(data)

                num = data.size(0)
                if self.args.dataset != 'face_verification':
                    train_accuracy.update(classification_accuracy(output, target), num)

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

    #{{{ Evaluate classification
    def validate(self, epoch_idx, biases=None):
        """Performs evaluation."""
        self.pruner.apply_mask()
        self.model.eval()
        val_loss = Metric('val_loss')
        val_accuracy = Metric('val_accuracy')

        with tqdm(total=len(self.val_loader),
                  desc='Validate Epoch  #{}: '.format(epoch_idx + 1),
                  ascii=True) as t:
            with torch.no_grad():
                for data, target in self.val_loader:
                    if self.args.cuda:
                        data, target = data.cuda(), target.cuda()

                    output = self.model(data)
                    num = data.size(0)
                    val_loss.update(self.criterion(output, target), num)
                    val_accuracy.update(classification_accuracy(output, target), num)

                    t.set_postfix({'loss': val_loss.avg.item(),
                                   'accuracy': '{:.2f}'.format(100. * val_accuracy.avg.item()),
                                   'sparsity': self.pruner.calculate_sparsity(),
                                   'task{} ratio'.format(self.inference_dataset_idx): self.pruner.calculate_curr_task_ratio(),
                                   'zero ratio': self.pruner.calculate_zero_ratio()})
                    t.update(1)
        return val_accuracy.avg.item()
    #}}}

    #{{{ Evaluate LFW
    def evalLFW(self, epoch_idx):
        distance_metric = True
        subtract_mean   = False
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
        print('Test set: Accuracy: {:.5f}+-{:.5f}'.format(np.mean(accuracy),np.std(accuracy)))
        return np.mean(accuracy)
    #}}}

    def one_shot_prune(self, one_shot_prune_perc):
        self.pruner.one_shot_prune(one_shot_prune_perc)
        return

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
            elif isinstance(module, nn.PReLU):
                self.shared_layer_info[self.args.dataset][
                    'prelu_layer_weight'][name] = module.weight

        checkpoint = {
            'model_state_dict': self.model.module.state_dict(),
            'dataset_history': self.model.module.datasets,
            'dataset2num_classes': self.model.module.dataset2num_classes,
            'masks': self.pruner.masks,
            'shared_layer_info': self.shared_layer_info,
            # 'optimizer_network_state_dict': optimizers[0].state_dict(),
        }

        torch.save(checkpoint, filepath)
        return

    def load_checkpoint(self, optimizers, resume_from_epoch, save_folder):

        if resume_from_epoch > 0:
            filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=resume_from_epoch)
            checkpoint = torch.load(filepath)
            checkpoint_keys = checkpoint.keys()
            state_dict = checkpoint['model_state_dict']
            curr_model_state_dict = self.model.module.state_dict()
            for name, param in state_dict.items():
                if (name == 'classifier.weight' or name == 'classifier.bias' or
                    (name == 'classifier.0.weight' or name == 'classifier.0.bias' or name == 'classifier.1.weight')):
                     # I DONT WANT TO DO THIS! QQ That last 3 exprs are for anglelinear and embeddings
                    continue
                else:
                    curr_model_state_dict[name].copy_(param)
        return

    def load_checkpoint_for_inference(self, resume_from_epoch, save_folder):

        if resume_from_epoch > 0:
            filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=resume_from_epoch)
            checkpoint = torch.load(filepath)
            checkpoint_keys = checkpoint.keys()
            state_dict = checkpoint['model_state_dict']
            curr_model_state_dict = self.model.module.state_dict()

            for name, param in state_dict.items():
                if (name == 'classifier.weight' or name == 'classifier.bias' or
                    (name == 'classifier.0.weight' or name == 'classifier.0.bias' or name == 'classifier.1.weight')):
                     # I DONT WANT TO DO THIS! QQ That last 3 exprs are for anglelinear and embeddings
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
                elif isinstance(module, nn.PReLU):
                    module.weight = self.shared_layer_info[self.args.dataset][
                        'prelu_layer_weight'][name]
