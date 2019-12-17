"""Contains a bunch of utility functions."""
import numpy as np
import pdb


def set_dataset_paths(args):
    """Set default train and test path if not provided as input."""
    args.train_path = 'data/%s/train' % (args.dataset)
    args.val_path   = 'data/%s/val'   % (args.dataset)
