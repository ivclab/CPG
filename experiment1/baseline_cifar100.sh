#!/bin/bash


DATASETS=(
    'None'                # dummy
    'aquatic_mammals'
    'fish'
    'flowers'
    'food_containers'
    'fruit_and_vegetables'
    'household_electrical_devices'
    'household_furniture'
    'insects'
    'large_carnivores'
    'large_man-made_outdoor_things'
    'large_natural_outdoor_scenes'
    'large_omnivores_and_herbivores'
    'medium_mammals'
    'non-insect_invertebrates'
    'people'
    'reptiles'
    'small_mammals'
    'trees'
    'vehicles_1'
    'vehicles_2'
)

GPU_ID=0
ARCH='vgg16_bn_cifar100'
FINETUNE_EPOCHS=100

for TASK_ID in `seq 1 20`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_cifar100_main_normal.py \
        --arch $ARCH \
        --dataset ${DATASETS[TASK_ID]} --num_classes 5 \
        --lr 1e-2 \
        --weight_decay 4e-5 \
        --save_folder checkpoints/baseline/experiment1/$ARCH/${DATASETS[TASK_ID]} \
        --epochs $FINETUNE_EPOCHS \
        --mode finetune \
        --logfile logs/baseline_cifar100_acc.txt
done

for HISTORY_ID in `seq 1 20`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_cifar100_main_normal.py \
        --arch $ARCH \
        --dataset ${DATASETS[HISTORY_ID]} --num_classes 5 \
        --load_folder checkpoints/baseline/experiment1/$ARCH/${DATASETS[HISTORY_ID]} \
        --mode inference
done
