# Usage: bash experiment1/baseline_cifar100.sh

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

GPU_ID=3
arch='vgg16_bn_cifar100'
finetune_epochs=100

for task_id in `seq 1 20`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_cifar100_main_normal.py \
        --arch $arch \
        --dataset ${dataset[task_id]} --num_classes 5 \
        --lr 1e-2 \
        --weight_decay 4e-5 \
        --save_folder checkpoints/baseline/experiment1/$arch/${dataset[task_id]} \
        --epochs $finetune_epochs \
        --mode finetune \
        --logfile logs/baseline_cifar100_acc.txt
done

# for history_id in `seq 1 20`; do
#     CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_cifar100_main_normal.py \
#         --arch $arch \
#         --dataset ${dataset[history_id]} --num_classes 5 \
#         --load_folder checkpoints/baseline/experiment1_1/$arch/${dataset[history_id]} \
#         --mode inference
# done
