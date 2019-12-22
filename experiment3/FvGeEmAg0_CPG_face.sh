#!/bin/bash
# Normally, bash shell cannot support floating point arthematic, thus, here we use `bc` package

PREFIX=FvGeEmAg0
TARGET_TASK_ID=4

dataset=(
      'None'                # dummy
		  'face_verification'
		  'gender'
		  'emotion'
      'age0'
)

num_classes=(
      0                     # dummy
      4630
	    3
	    7
      8
)

init_lr=(
    0                       # dummy
    1e-3
    5e-4
    5e-4
    5e-4
)


batch_size=(
    0                       # dummy
    256
    128
    128
    128
)


finetune_start_sparsity=(
    0                       # dummy
    0
    0.5
    0.1
    0.2
)


acc_margin=(
    0                       # dummy
    0.005
    0.015
    0.005
    0.010
)


GPU_ID=0,1,2,3
arch='spherenet20'
finetune_epochs=100
network_width_multiplier=1.0
pruning_ratio_interval=0.1


for task_id in `seq $TARGET_TASK_ID $TARGET_TASK_ID`; do
    state=2
    while [ $state -eq 2 ]; do
        if [ "$task_id" != "1" ]
        then
            CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_face_main.py \
                --arch $arch \
                --dataset ${dataset[task_id]} --num_classes ${num_classes[task_id]} \
                --lr ${init_lr[task_id]} \
                --lr_mask 5e-4 \
                --weight_decay 4e-5 \
                --save_folder checkpoints/CPG/experiment3/$arch/${dataset[task_id]}/scratch \
                --load_folder checkpoints/CPG/experiment3/$arch/${dataset[task_id-1]}/gradual_prune/${finetune_start_sparsity[task_id]} \
                --epochs $finetune_epochs \
                --mode finetune \
                --batch_size ${batch_size[task_id]} \
                --val_batch_size 1 \
                --acc_margin ${acc_margin[task_id]} \
                --network_width_multiplier $network_width_multiplier \
                --jsonfile logs/baseline_face_acc.txt \
                --log_path checkpoints/CPG/experiment3/$arch/run.log
        else
            CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_face_main.py \
                --arch $arch \
                --dataset ${dataset[task_id]} --num_classes ${num_classes[task_id]} \
                --lr ${init_lr[task_id]} \
                --lr_mask 5e-4 \
                --weight_decay 4e-5 \
                --save_folder checkpoints/CPG/experiment3/$arch/${dataset[task_id]}/scratch \
                --epochs $finetune_epochs \
                --mode finetune \
                --batch_size ${batch_size[task_id]} \
                --val_batch_size 1 \
                --acc_margin ${acc_margin[task_id]} \
                --network_width_multiplier $network_width_multiplier \
                --jsonfile logs/baseline_face_acc.txt \
                --use_vgg_pretrained \
                --log_path checkpoints/CPG/experiment3/$arch/run.log
        fi

        state=$?
        if [ $state -eq 2 ]
        then
            network_width_multiplier=$(bc <<< $network_width_multiplier+0.5)
            echo "New network_width_multiplier: $network_width_multiplier"
            continue
        elif [ $state -eq 3 ]
        then
            echo "You should provide the baseline_face_acc.txt"
            exit 0
        fi

        nrof_epoch=0
        if [ "$task_id" == "1" ]
        then
          nrof_epoch_for_each_prune=10
          pruning_frequency=1000
        else
          nrof_epoch_for_each_prune=20
          pruning_frequency=100
        fi
        start_sparsity=0.0
        end_sparsity=0.1
        nrof_epoch=$nrof_epoch_for_each_prune

        # gradually pruning
        CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_face_main.py \
            --arch $arch \
            --dataset ${dataset[task_id]} --num_classes ${num_classes[task_id]} \
            --lr 0.0005 \
            --lr_mask 0.0 \
            --weight_decay 4e-5 \
            --save_folder checkpoints/CPG/experiment3/$arch/${dataset[task_id]}/gradual_prune/$end_sparsity \
            --load_folder checkpoints/CPG/experiment3/$arch/${dataset[task_id]}/scratch \
            --epochs $nrof_epoch \
            --mode prune \
            --initial_sparsity=$start_sparsity \
            --target_sparsity=$end_sparsity \
            --pruning_frequency=$pruning_frequency \
            --pruning_interval=4 \
            --jsonfile logs/baseline_face_acc.txt \
            --batch_size ${batch_size[task_id]}\
            --val_batch_size 1 \
            --acc_margin ${acc_margin[task_id]} \
            --network_width_multiplier $network_width_multiplier \
            --log_path checkpoints/CPG/experiment3/$arch/run.log
        state=$?
        if [ $state -eq 2 ]
        then
            network_width_multiplier=$(bc <<< $network_width_multiplier+0.5)
            echo "New network_width_multiplier: $network_width_multiplier"
            continue
        fi
    done

    if [ $state -eq 4 ]
    then
        continue
    fi

    for RUN_ID in `seq 1 9`; do
        nrof_epoch=$nrof_epoch_for_each_prune
        start_sparsity=$end_sparsity
        if [ $RUN_ID -lt 9 ]
        then
            end_sparsity=$(printf "%.1f" $(bc <<< $end_sparsity+$pruning_ratio_interval))
        else
            end_sparsity=$(printf "%.2f" $(bc <<< $end_sparsity+0.05))
        fi

        CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_face_main.py \
            --arch $arch \
            --dataset ${dataset[task_id]} --num_classes ${num_classes[task_id]} \
            --lr 0.0005 \
            --lr_mask 0.0 \
            --weight_decay 4e-5 \
            --save_folder checkpoints/CPG/experiment3/$arch/${dataset[task_id]}/gradual_prune/$end_sparsity \
            --load_folder checkpoints/CPG/experiment3/$arch/${dataset[task_id]}/gradual_prune/$start_sparsity \
            --epochs $nrof_epoch \
            --mode prune \
            --initial_sparsity=$start_sparsity \
            --target_sparsity=$end_sparsity \
            --pruning_frequency=$pruning_frequency \
            --pruning_interval=4 \
            --batch_size ${batch_size[task_id]} \
            --val_batch_size 1 \
            --jsonfile logs/baseline_face_acc.txt \
            --acc_margin ${acc_margin[task_id]} \
            --network_width_multiplier $network_width_multiplier \
            --log_path checkpoints/CPG/experiment3/$arch/run.log
        if [ $? -eq 4 ]
        then
            break
        fi
    done
done
