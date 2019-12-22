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

INIT_LR=(
    0
    1e-3
    1e-3
    1e-2
    1e-3
    1e-3
    1e-3
)

GPU_ID=0,1,2,3
ARCH='resnet50'
FINETUNE_EPOCHS=100

# ResNet50 pretrained on ImageNet
echo {\"imagenet\": \"0.7616\"} > logs/baseline_imagenet_acc_${ARCH}.txt

for TASK_ID in `seq 2 6`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_imagenet_main.py \
        --arch $ARCH \
        --dataset ${DATASETS[TASK_ID]} --num_classes ${NUM_CLASSES[TASK_ID]} \
        --lr ${INIT_LR[TASK_ID]} \
        --weight_decay 4e-5 \
        --save_folder checkpoints/baseline/experiment2/$ARCH/${DATASETS[TASK_ID]} \
        --epochs $FINETUNE_EPOCHS \
        --mode finetune \
        --logfile logs/baseline_imagenet_acc_${ARCH}.txt \
        --use_imagenet_pretrained
done
