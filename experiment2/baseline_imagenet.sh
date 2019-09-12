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

GPU_ID=0
# arch='vgg16_bn'
arch='resnet50'

finetune_epochs=100

for task_id in `seq 2 6`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python packnet_imagenet_main.py \
        --arch $arch \
        --dataset ${dataset[task_id]} --num_classes ${num_classes[task_id]} \
        --lr ${init_lr[task_id]} \
        --weight_decay 4e-5 \
        --save_folder checkpoints/baseline/$arch/${dataset[task_id]} \
        --epochs $finetune_epochs \
        --mode finetune \
        --logfile logs/baseline_imagenet_acc_$arch.txt \
        --use_imagenet_pretrained
done
