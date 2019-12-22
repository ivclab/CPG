#!/bin/bash


TARGET_TASK_ID=1

dataset=(
    'None'     # dummy
    'imagenet'
    'cubs_cropped'
    'stanford_cars_cropped'
    'flowers'
    'wikiart'
    'sketches'
)

num_classes=(
    0
    1000
    200
    196
    102
    195
    250
)

init_lr=(
    0
    1e-3
    1e-3
    1e-2
    1e-3
    1e-3
    1e-3
)

pruning_lr=(
    0
    3e-4
    1e-3
    1e-3
    1e-3
    1e-3
    1e-3
)

GPU_ID=0,1,2,3,4,5,6,7
arch='resnet50'
finetune_epochs=300
network_width_multiplier=1.0
pruning_ratio_interval=0.1
lr_mask=1e-4


for task_id in `seq $TARGET_TASK_ID $TARGET_TASK_ID`; do
    state=2
    while [ $state -eq 2 ]; do
        if [ "$task_id" != "1" ]
        then
            CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_imagenet_main.py \
               --arch $arch \
               --dataset ${dataset[task_id]} --num_classes ${num_classes[task_id]} \
               --lr ${init_lr[task_id]} \
               --lr_mask $lr_mask \
               --weight_decay 4e-5 \
               --save_folder checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/scratch \
               --load_folder checkpoints/CPG/experiment2/$arch/${dataset[task_id-1]}/gradual_prune \
               --epochs $finetune_epochs \
               --mode finetune \
               --network_width_multiplier $network_width_multiplier \
               --pruning_ratio_to_acc_record_file checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/gradual_prune/record.txt \
               --jsonfile logs/baseline_imagenet_acc_$arch.txt \
               --log_path checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/train.log
        else
            CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_imagenet_main.py \
               --arch $arch \
               --dataset ${dataset[task_id]} --num_classes ${num_classes[task_id]} \
               --lr ${init_lr[task_id]} \
               --weight_decay 4e-5 \
               --save_folder checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/scratch \
               --epochs $finetune_epochs \
               --mode finetune \
               --network_width_multiplier $network_width_multiplier \
               --jsonfile logs/baseline_imagenet_acc_$arch.txt \
               --pruning_ratio_to_acc_record_file checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/gradual_prune/record.txt \
               --use_imagenet_pretrained
        fi

        state=$?
        if [ $state -eq 2 ]
        then
            network_width_multiplier=$(bc <<< $network_width_multiplier+0.5)
            echo "New network_width_multiplier: $network_width_multiplier"
            break
            continue
        elif [ $state -eq 3 ]
        then
            echo "You should provide the baseline_cifar100_acc.txt as criterion to decide whether the capacity of network is enough for new task"
            exit 0
        fi
    done

    nrof_epoch=0
    if [ "$task_id" == "1" ]
    then
        nrof_epoch_for_each_prune=10
        pruning_frequency=1000
    else
        nrof_epoch_for_each_prune=20
        pruning_frequency=50
    fi
    start_sparsity=0.0
    end_sparsity=0.1
    nrof_epoch=$nrof_epoch_for_each_prune

    if [ $state -ne 5 ]
    then
        # gradually pruning
        CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_imagenet_main.py \
            --arch $arch \
            --dataset ${dataset[task_id]} --num_classes ${num_classes[task_id]}  \
            --lr ${pruning_lr[task_id]} \
            --lr_mask 0.0 \
            --weight_decay 4e-5 \
            --save_folder checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/gradual_prune \
            --load_folder checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/scratch \
            --epochs $nrof_epoch \
            --mode prune \
            --initial_sparsity=$start_sparsity \
            --target_sparsity=$end_sparsity \
            --pruning_frequency=$pruning_frequency \
            --pruning_interval=4 \
            --jsonfile logs/baseline_imagenet_acc_$arch.txt \
            --network_width_multiplier $network_width_multiplier \
            --pruning_ratio_to_acc_record_file checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/gradual_prune/record.txt \
            --log_path checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/train.log

        for RUN_ID in `seq 1 9`; do
            nrof_epoch=$nrof_epoch_for_each_prune
            start_sparsity=$end_sparsity
            if [ $RUN_ID -lt 9 ]
            then
                end_sparsity=$(bc <<< $end_sparsity+$pruning_ratio_interval)
            else
                end_sparsity=$(bc <<< $end_sparsity+0.05)
            fi

            CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_imagenet_main.py \
                --arch $arch \
                --dataset ${dataset[task_id]} --num_classes ${num_classes[task_id]} \
                --lr ${pruning_lr[task_id]} \
                --lr_mask 0.0 \
                --weight_decay 4e-5 \
                --save_folder checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/gradual_prune \
                --load_folder checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/gradual_prune \
                --epochs $nrof_epoch \
                --mode prune \
                --initial_sparsity=$start_sparsity \
                --target_sparsity=$end_sparsity \
                --pruning_frequency=$pruning_frequency \
                --pruning_interval=4 \
                --jsonfile logs/baseline_imagenet_acc_$arch.txt \
                --network_width_multiplier $network_width_multiplier \
                --pruning_ratio_to_acc_record_file checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/gradual_prune/record.txt \
                --log_path checkpoints/CPG/experiment2/$arch/${dataset[task_id]}/train.log
        done
    fi
done
