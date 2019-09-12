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
    100)


GPU_ID=1 #,1,2,3,4,5,6,7
network_width_multiplier=1.0
arch='custom_vgg'
#arch='resnet50'

for task_id in `seq 2 5`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python CPG_imagenet_main.py \
        --arch $arch \
        --dataset ${dataset[$task_id]} --num_classes ${num_classes[task_id]} \
        --load_folder checkpoints/ProgressiveNet/$arch/${dataset[task_id]}/scratch \
        --mode inference \
        --jsonfile logs/baseline_imagenet_acc_$arch.txt \
        --network_width_multiplier $network_width_multiplier \
        --log_path logs/ProgressiveNet_$arch_${dataset[task_id]}_val.log
done
