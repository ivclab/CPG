#!/bin/bash
# Usage:
#   bash scripts/inference_FvGeEmAg.sh FvGeEmAg0 0 0.2 age0 logs/FvGeEmAg0_s02.log
#   bash scripts/inference_FvGeEmAg.sh FvGeEmAg1 0 0.3 age1 logs/FvGeEmAg1_s03.log
#   bash scripts/inference_FvGeEmAg.sh FvGeEmAg2 0 0.1 age2 logs/FvGeEmAg2_s01.log
#   bash scripts/inference_FvGeEmAg.sh FvGeEmAg3 0 0.1 age3 logs/FvGeEmAg3_s01.log
#   bash scripts/inference_FvGeEmAg.sh FvGeEmAg4 0 0.2 age4 logs/FvGeEmAg4_s02.log

PREFIX=$1
GPU_ID=$2
TARGET_SPARSITY=$3
AGEFOLD=$4
LOG_PATH=$5

DATASET=(
  'face_verification'
  'gender'
  'emotion'
)

NUM_CLASSES=(
  4630
  3
  7
)

ARCH='spherenet20'
NETWORK_WIDTH_MULTIPLIER=1.0
SPARSITY_DIR=${PREFIX}_checkpoints/CPG/$ARCH/$AGEFOLD/gradual_prune/$TARGET_SPARSITY


echo "In directory: " $SPARSITY_DIR
for task_id in `seq 0 2`; do
  CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_face_main.py \
      --arch $ARCH \
      --dataset ${DATASET[task_id]} --num_classes ${NUM_CLASSES[task_id]} \
      --load_folder $SPARSITY_DIR \
      --mode inference \
      --jsonfile logs/baseline_face_acc.txt \
      --log_path $LOG_PATH \
      --network_width_multiplier $NETWORK_WIDTH_MULTIPLIER
done


CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_face_main.py \
    --arch $ARCH \
    --dataset $AGEFOLD --num_classes 8 \
    --load_folder $SPARSITY_DIR \
    --mode inference \
    --jsonfile logs/baseline_face_acc.txt \
    --log_path $LOG_PATH \
    --network_width_multiplier $NETWORK_WIDTH_MULTIPLIER
