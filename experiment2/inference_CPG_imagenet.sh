#!/bin/bash


DATASETS=(
    'None'     # dummy
    'imagenet'
    'cubs_cropped'
    'stanford_cars_cropped'
    'flowers'
    'wikiart'
    'sketches'
)

NUM_CLASSES=(
    0
    1000
    200
    196
    102
    195
    250
)

GPU_ID=3
NETWORK_WIDTH_MULTIPLIER=1.0
ARCH='resnet50'


for TASK_ID in `seq 1 6`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_imagenet_main.py \
        --arch $ARCH \
        --dataset ${DATASETS[$TASK_ID]} --num_classes ${NUM_CLASSES[TASK_ID]} \
        --load_folder checkpoints/CPG/experiment2/$ARCH/${DATASETS[6]}/gradual_prune \
        --mode inference \
        --jsonfile logs/baseline_imagenet_acc_$ARCH.txt \
        --network_width_multiplier $NETWORK_WIDTH_MULTIPLIER \
        --log_path logs/imagenet_inference.log
done
