import argparse
import json
import pdb
import os
import sys
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--save_folder', type=str, help='')
parser.add_argument('--load_folder', type=str, help='')

if __name__ == '__main__':
    args = parser.parse_args()
    src_filenames = os.listdir(args.load_folder)
    for src_filename in src_filenames:
        if '.pth.tar' in src_filename:
            out_paths = os.listdir(args.save_folder)
            for checkpoint_file in out_paths:
                if '.pth.tar' in checkpoint_file:
                    os.remove(os.path.join(args.save_folder, checkpoint_file))
            
            shutil.copyfile(os.path.join(args.load_folder, src_filename), os.path.join(args.save_folder, src_filename))
            break
 
