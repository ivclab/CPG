# choose the checkpoint file that we want as an initial one for next task
import shutil
import argparse
import json
import pdb
import os
import logging
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--pruning_ratio_to_acc_record_file', type=str, help='')
parser.add_argument('--allow_acc_loss', type=float, default=0.0, help='')
parser.add_argument('--baseline_acc_file', type=str, help='')
parser.add_argument('--dataset', type=str, help='')
parser.add_argument('--network_width_multiplier', type=float, help='')
parser.add_argument('--max_allowed_network_width_multiplier', type=float, help='')
parser.add_argument('--log_path', type=str, help='')

def set_logger(filepath):
    global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    _format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return


def main():
    args = parser.parse_args()
    if args.log_path:
        set_logger(args.log_path)

    save_folder = args.pruning_ratio_to_acc_record_file.rsplit('/', 1)[0]
    with open(args.baseline_acc_file, 'r') as jsonfile:
        json_data = json.load(jsonfile)
        criterion_acc = float(json_data[args.dataset])

    with open(args.pruning_ratio_to_acc_record_file, 'r') as json_file:
        json_data = json.load(json_file)
        acc_before_prune = json_data['0.0']
        json_data.pop('0.0')
        available_pruning_ratios = list(json_data.keys())
        available_pruning_ratios.reverse()
        flag_there_is_pruning_ratio_that_match_our_need = False

        chosen_pruning_ratio = 0.0
        for pruning_ratio in available_pruning_ratios:
            acc = json_data[pruning_ratio]
            #criterion_acc = min(criterion_acc, acc_before_prune)
            if (acc + args.allow_acc_loss >= criterion_acc) or (
                (args.network_width_multiplier == args.max_allowed_network_width_multiplier) and (acc_before_prune < criterion_acc)):
                chosen_pruning_ratio = pruning_ratio
                checkpoint_folder = os.path.join(save_folder, str(pruning_ratio))

                for filename in os.listdir(checkpoint_folder):
                    shutil.copyfile(os.path.join(checkpoint_folder, filename), os.path.join(save_folder, filename))
                flag_there_is_pruning_ratio_that_match_our_need = True
                break

        for pruning_ratio in available_pruning_ratios:
            checkpoint_folder = os.path.join(save_folder, str(pruning_ratio))
            shutil.rmtree(checkpoint_folder)

        if not flag_there_is_pruning_ratio_that_match_our_need:
            folder_that_contain_checkpoint_before_pruning = os.path.join(save_folder.rsplit('/', 1)[0], 'scratch')
            for filename in os.listdir(folder_that_contain_checkpoint_before_pruning):
                shutil.copyfile(os.path.join(folder_that_contain_checkpoint_before_pruning, filename), os.path.join(save_folder, filename))

        logging.info('We choose pruning_ratio {}'.format(chosen_pruning_ratio))

if __name__ == '__main__':
  main()
