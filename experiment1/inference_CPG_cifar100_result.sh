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

GPU_ID=0
network_width_multiplier=1.0
arch='custom_vgg_cifar100'

setting='finetune_max_mul_1.5'

for task_id in `seq 2 2`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_cifar100_with_one_mask.py \
        --arch $arch \
        --dataset ${dataset[$task_id]} --num_classes 5 \
        --load_folder checkpoints/CPG/$setting/$arch/${dataset[2]}/gradual_prune \
        --mode inference \
        --baseline_acc_file logs/finetune_max/ \
        --network_width_multiplier $network_width_multiplier \
        --max_allowed_network_width_multiplier $max_allowed_network_width_multiplier 1.5 \
        --log_path checkpoints/CPG/$setting/inference.log
done
        # --load_folder checkpoints/PAE_adam/$arch/${dataset[3]}/gradual_prune \
