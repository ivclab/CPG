dataset=('None'     # dummy
         'imagenet'
         'cubs_cropped'
         'stanford_cars_cropped'
         'flowers'
         'wikiart'
         'sketches')

num_classes=(
    0
    1000
    200
    196
    102
    195
    250
)


GPU_ID=3 #,1,2,3,4,5,6,7
network_width_multiplier=1.0
arch='resnet50'

pruning_ratio=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95)

for task_id in `seq 5 5`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_imagenet_main.py \
        --arch $arch \
        --dataset ${dataset[$task_id]} --num_classes ${num_classes[task_id]} \
        --load_folder checkpoints/CPG/$arch/${dataset[5]}/gradual_prune \
        --mode inference \
        --jsonfile logs/baseline_imagenet_acc_$arch.txt \
        --network_width_multiplier $network_width_multiplier
        #--log_path checkpoints/CPG/$arch/${dataset[3]}_3/val_2.log

   # for idx in `seq 0 9`; do
   #     CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_imagenet_main.py \
   #         --arch $arch \
   #         --dataset ${dataset[$task_id]} --num_classes ${num_classes[task_id]} \
   #         --load_folder checkpoints/CPG/$arch/${dataset[3]}_3/gradual_prune/${pruning_ratio[idx]} \
   #         --mode inference \
   #         --jsonfile logs/baseline_imagenet_acc_$arch.txt \
   #         --network_width_multiplier $network_width_multiplier \
   #         --log_path checkpoints/CPG/$arch/${dataset[3]}_3/val_2.log
   # done
done
