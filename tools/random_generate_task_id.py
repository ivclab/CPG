import random
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--curr_task_id', type=int, help='')
    args = parser.parse_args()
    sys.exit(random.randint(1, args.curr_task_id-1))
