dataset=('None'     # dummy
         'imagenet'
         'cubs_cropped'
         'stanford_cars_cropped'
         'flowers'
         'wikiart'
         'sketches'
         'dtd'
	 'fgvc')

num_classes=(
    0
    1000
    200
    196
    102
    195
    250
    47
    100
)

init_lr=(
    0
    1e-3
    1e-3
    1e-2
    1e-3
    1e-3
    1e-3
    1e-3
    1e-2
)

GPU_ID=1   #0,1,2,3,4,5,6,7
arch='resnet50'
task_id=1
finetune_epochs=100
network_width_multiplier=6.5 # TODO
pruning_ratio_interval=0.1

# adjust lr_mask, lr

# inference imagenet vgg16bn baseline
#CUDA_VISIBLE_DEVICES=$GPU_ID python PAE_imagenet_main.py \
#    --arch $arch \
#    --dataset ${dataset[$task_id]} --num_classes ${num_classes[task_id]} \
#    --load_folder checkpoints/ProgressiveNet/$arch/${dataset[1]}/scratch \
#    --mode inference \
#    --jsonfile logs/baseline_imagenet_acc_$arch.txt \
#    --network_width_multiplier 1.0

for task_id in `seq 7 8`; do
    state=2
    while [ $state -eq 2 ]; do
        if [ "$task_id" != "1" ]
        then
            CUDA_VISIBLE_DEVICES=$GPU_ID python PAE_imagenet_main_10iter.py \
                --arch $arch \
                --dataset ${dataset[task_id]} --num_classes ${num_classes[task_id]} \
                --lr ${init_lr[task_id]} \
                --lr_mask 0.0 \
                --weight_decay 4e-5 \
                --save_folder checkpoints/ProgressiveNet/$arch/${dataset[task_id]}/scratch \
                --load_folder checkpoints/ProgressiveNet/$arch/${dataset[task_id-1]}/scratch \
                --epochs $finetune_epochs \
                --mode finetune \
                --network_width_multiplier $network_width_multiplier \
                --jsonfile logs/baseline_imagenet_acc_$arch.txt
        else
            :
            # CUDA_VISIBLE_DEVICES=$GPU_ID python PAE_imagenet_main.py \
            #    --arch $arch \
            #    --dataset ${dataset[task_id]} --num_classes ${num_classes[task_id]} \
            #    --lr ${init_lr[task_id]} \
            #    --lr_mask 5e-4 \
            #    --weight_decay 4e-5 \
            #    --save_folder checkpoints/ProgressiveNet/$arch/${dataset[task_id]}/scratch \
            #    --epochs $finetune_epochs \
            #    --mode finetune \
            #    --network_width_multiplier $network_width_multiplier \
            #    --jsonfile logs/baseline_imagenet_acc.txt \
            #    --use_imagenet_pretrained
        fi

        state=$?
        if [ $state -eq 2 ]
        then
            network_width_multiplier=$(bc <<< $network_width_multiplier+0.5)        
            echo "New network_width_multiplier: $network_width_multiplier"
            bool2=$(echo "$network_width_multiplier == 2.5" | bc)
            bool3=$(echo "$network_width_multiplier == 3.5" | bc)
            bool4=$(echo "$network_width_multiplier == 4.5" | bc)
            bool5=$(echo "$network_width_multiplier == 5.5" | bc)
            bool6=$(echo "$network_width_multiplier == 6.5" | bc)
	    bool7=$(echo "$network_width_multiplier == 7.5" | bc)
	    bool8=$(echo "$network_width_multiplier == 8.5" | bc)
            if [ "$bool2" -eq 1 -o "$bool3" -eq 1 -o "$bool4" -eq 1 -o "$bool5" -eq 1 -o "$bool6" -eq 1 -o "$bool7" -eq 1 -o "$bool8" -eq 1 ]
	    then
		break
	    else	
            	continue
	    fi	
        elif [ $state -eq 3 ]
        then
            echo "You should provide the baseline_cifar100_acc.txt as criterion to decide whether the capacity of network is enough for new task"
            exit 0
        fi
    done
done
