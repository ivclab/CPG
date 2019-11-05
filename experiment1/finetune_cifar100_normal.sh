dataset=('None'                # dummy
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
         'vehicles_2')

GPU_ID=1
arch='vgg16_bn_cifar100'
finetune_epochs=100

initial_from_task_id=0
for task_id in `seq 1 20`; do
    if [ "$task_id" != "1" ]
    then
        echo "Start training task " $task_id
        python TOOLS/random_generate_task_id.py --curr_task_id $task_id
        initial_from_task_id=$?
        echo "Initial for curr task is " $initial_from_task_id
        CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_cifar100_main_normal.py \
            --arch $arch \
            --dataset ${dataset[task_id]} --num_classes 5 \
            --lr 5e-3 \
            --weight_decay 4e-5 \
            --save_folder checkpoints/finetune/experiment1/$arch/${dataset[task_id]} \
            --epochs $finetune_epochs \
            --mode finetune \
            --logfile logs/finetune_cifar100_acc_normal_5e-3.txt \
            --initial_from_task checkpoints/finetune/experiment1/$arch/${dataset[initial_from_task_id]}
    else
        echo "Start training task ", $task_id
        CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_cifar100_main_normal.py \
            --arch $arch \
            --dataset ${dataset[task_id]} --num_classes 5 \
            --lr 1e-2 \
            --weight_decay 4e-5 \
            --save_folder checkpoints/finetune/experiment1/$arch/${dataset[task_id]} \
            --epochs $finetune_epochs \
            --mode finetune \
            --logfile logs/finetune_cifar100_acc_normal_5e-3.txt \
            --initial_from_task checkpoints/finetune/experiment1/$arch/${dataset[initial_from_task_id]}
    fi
done

for history_id in `seq 1 20`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_cifar100_main_normal.py \
        --arch $arch \
        --dataset ${dataset[history_id]} --num_classes 5 \
        --load_folder checkpoints/finetune/experiment1/$arch/${dataset[history_id]} \
        --mode inference
done
