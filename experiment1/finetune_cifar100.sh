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

GPU_ID=1
ARCH='vgg16_bn_cifar100'
FINETUNE_EPOCHS=100
LR=5e-3
ITER_TIMES=5

for ITER_ID in `seq 1 $ITER_TIMES`; do

    INITIAL_FROM_TASK_ID=0
    for TASK_ID in `seq 1 20`; do
        if [ "$TASK_ID" != "1" ]
        then
            echo "Start training task " $TASK_ID
            python tools/random_generate_task_id.py --curr_task_id $TASK_ID
            INITIAL_FROM_TASK_ID=$?
            echo "Initial for curr task is " $INITIAL_FROM_TASK_ID
            CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_cifar100_main_normal.py \
                --arch $ARCH \
                --dataset ${DATASETS[TASK_ID]} --num_classes 5 \
                --lr $LR \
                --weight_decay 4e-5 \
                --save_folder checkpoints/finetune/experiment1_${ITER_ID}/$ARCH/${DATASETS[TASK_ID]} \
                --epochs $FINETUNE_EPOCHS \
                --mode finetune \
                --logfile logs/finetune_cifar100_acc_${ITER_ID}.txt \
                --initial_from_task checkpoints/finetune/experiment1_${ITER_ID}/$ARCH/${DATASETS[INITIAL_FROM_TASK_ID]}
        else
            echo "Start training task ", $TASK_ID
            CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_cifar100_main_normal.py \
                --arch $ARCH \
                --dataset ${DATASETS[TASK_ID]} --num_classes 5 \
                --lr 1e-2 \
                --weight_decay 4e-5 \
                --save_folder checkpoints/finetune/experiment1_${ITER_ID}/$ARCH/${DATASETS[TASK_ID]} \
                --epochs $FINETUNE_EPOCHS \
                --mode finetune \
                --logfile logs/finetune_cifar100_acc_${ITER_ID}.txt \
                --initial_from_task checkpoints/finetune/experiment1_${ITER_ID}/$ARCH/${DATASETS[INITIAL_FROM_TASK_ID]}
        fi
    done

    for HISTORY_ID in `seq 1 20`; do
        CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_cifar100_main_normal.py \
            --arch $ARCH \
            --dataset ${DATASETS[HISTORY_ID]} --num_classes 5 \
            --load_folder checkpoints/finetune/experiment1_${ITER_ID}/$ARCH/${DATASETS[HISTORY_ID]} \
            --mode inference
    done

done
