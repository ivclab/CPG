#!/bin/bash


DATASETS=(
    'None'                   # dummy
    'face_verification'
    'emotion'
    'gender'
    'age0'
    'age1'
    'age2'
    'age3'
    'age4'
)

NUM_CLASSES=(
      0                       # dummy
      4630
	    7
	    3
      8
      8
      8
      8
      8
)

INIT_LRS=(
      0.0                     # dummy
      1e-3
      1e-3
      1e-4
      1e-3
      1e-3
      1e-3
      1e-3
      1e-3
)

GPU_ID=0,1,2,3
ARCH='spherenet20'
FINETUNE_EPOCHS=100

# CNN20 pretrained on the face verification task
echo {\"face_verification\": \"0.9942\"} > logs/baseline_face_acc.txt

for TASK_ID in `seq 2 8`; do
      CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_face_main.py \
          --arch $ARCH \
          --dataset ${DATASETS[TASK_ID]} \
  		    --num_classes ${NUM_CLASSES[TASK_ID]} \
          --lr ${INIT_LRS[TASK_ID]} \
          --weight_decay 4e-5 \
  	      --batch_size 32 \
  	      --val_batch_size 1 \
          --save_folder checkpoints/baseline/experiment3/$ARCH/${DATASETS[TASK_ID]} \
          --epochs $FINETUNE_EPOCHS \
          --mode finetune \
          --logfile logs/baseline_face_acc.txt \
          --use_vgg_pretrained
done
