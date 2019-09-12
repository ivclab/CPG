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
    1e-3 # TODO
    1e-3 # TODO
    1e-3 # TODO
    1e-3 # TODO
    1e-3 # TODO
)

GPU_ID=0
arch='resnet50'
finetune_epochs=100
network_width_multiplier=1.0 # TODO
pruning_ratio_interval=0.1

# adjust lr_mask, lr

# 1e-4
# 5e-4
lr_mask=1e-4
# lr_mask=5e-4

# for task_id in `seq 2 6`; do
#     if [ "$task_id" != "1" ]
#     then
#         CUDA_VISIBLE_DEVICES=$GPU_ID python PAE_imagenet_main.py \
#             --arch $arch \
#             --dataset ${dataset[task_id]} --num_classes ${num_classes[task_id]} \
#             --lr 1e-4 \
#             --lr_mask $lr_mask \
#             --weight_decay 4e-5 \
#             --save_folder checkpoints/Piggyback/${arch}_${lr_mask}/${dataset[task_id]}/scratch \
#             --load_folder checkpoints/Piggyback/${arch}_${lr_mask}/${dataset[task_id-1]}/scratch \
#             --epochs $finetune_epochs \
#             --mode finetune \
#             --network_width_multiplier $network_width_multiplier \
#             --jsonfile logs/baseline_imagenet_acc_$arch.txt \
#             --test_piggymask
#     else
#         :
#         # CUDA_VISIBLE_DEVICES=$GPU_ID python PAE_imagenet_main.py \
#         #    --arch $arch \
#         #    --dataset ${dataset[task_id]} --num_classes ${num_classes[task_id]} \
#         #    --lr ${init_lr[task_id]} \
#         #    --weight_decay 4e-5 \
#         #    --save_folder checkpoints/Piggyback/${arch}_${lr_mask}/${dataset[task_id]}/scratch \
#         #    --epochs $finetune_epochs \
#         #    --mode finetune \
#         #    --network_width_multiplier $network_width_multiplier \
#         #    --jsonfile logs/baseline_imagenet_acc_$arch.txt \
#         #    --use_imagenet_pretrained
#     fi
# done

# inference
for task_id in `seq 2 6`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python PAE_imagenet_main.py \
        --arch $arch \
        --dataset ${dataset[task_id]} --num_classes ${num_classes[task_id]} \
        --load_folder checkpoints/Piggyback/${arch}_${lr_mask}/${dataset[6]}/scratch \
        --mode inference \
        --jsonfile logs/baseline_imagenet_acc_$arch.txt \
        --network_width_multiplier 1.0
done
