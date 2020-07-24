# Compacting, Picking and Growing (CPG)

This is an official Pytorch implementation of CPG - a lifelong learning algorithm for object classification. For details about CPG please refer to the paper [Compacting, Picking and Growing for Unforgetting Continual Learning](http://papers.nips.cc/paper/9518-compacting-picking-and-growing-for-unforgetting-continual-learning.pdf) ([Slides](https://github.com/ivclab/CPG/blob/master/docs/%5BSlides_PDF%5D19NeurIPS_unforgetting_continual_Learning.pdf),[Poster](https://github.com/ivclab/CPG/blob/master/docs/%5BPoster%5D19NeurIPS_unforgetting_continual_learning.pdf))

The code is released for academic research use only. For commercial use, please contact [Dr. Chu-Song Chen](https://www.iis.sinica.edu.tw/pages/song/)(song@iis.sinica.edu.tw).

## Citing Paper
Please cite following paper if these codes help your research:

    @inproceedings{hung2019compacting,
    title={Compacting, Picking and Growing for Unforgetting Continual Learning},
    author={Hung, Ching-Yi and Tu, Cheng-Hao and Wu, Cheng-En and Chen, Chien-Hung and Chan, Yi-Ming and Chen, Chu-Song},
    booktitle={Advances in Neural Information Processing Systems},
    pages={13647--13657},
    year={2019}
    }

---

## Dependencies
    Python>=3.6
    PyTorch>=1.0
    tqdm
---

## Experiment1 (Compact 20 tasks into VGG16 network)

**Step 1.** Download CIFAR100 and form the 20 tasks based on their super classes with the [cifar2png](https://github.com/knjcode/cifar2png) tool. Or you can just download the converted version of our CIFAR100 from [here](https://drive.google.com/file/d/1eo2RhMmhxzUNOZa0Z7jy7y4lOn3lqddU/view?usp=sharing). Unzip the compressed file and place `cifar100_org/` in `data/`. 

**Step 2.** Use the following command to train individual models for each of the 20 tasks so that we can obtain their accuracy goals. 

```
$ bash experiement1/baseline_cifar100.sh 
```

If you would like to use higher accuracy goals, execute `experiment1/finetune_cifar100.sh` instead. The script randomly selects a model trained on previous tasks and finetunes it to the current one. After this step, we obtain `logs/baseline_cifar100_acc.txt` that contains accuracy goals for 20 tasks. 


**Step 3.** Run CPG to learn 20 tasks sequentially.

```
$ bash experiment1/CPG_cifar100_scratch_mul_1.5.sh
```

If you use another accuracy goals, please modify the `baseline_cifar100_acc` variable in `experiment1/CPG_cifar100_scratch_mul_1.5.sh` to the path containing your accuracy goals. 


**Step 4.** Inference the learned 20 tasks. 

```
$ bash experiment1/inference_CPG_cifar100.sh
```


### CPG-VGG16 [Checkpoints](https://drive.google.com/file/d/1Zc4MJGPMcWSUkxw2j2Zy7s18jDUALkwc/view?usp=sharing) on CIFAR-100 Twenty Tasks.

Extract the downloaded .zip file and place `all_max_mul_1.5/` in `checkpoints/CPG/experiment1/`. Modify the `SETTING` variable in `experiment1/inference_CPG_cifar100.sh` to `all_max_mul_1.5` before inference. 


| Task |   1  |   2  |   3  |   4  |   5  |   6  |   7  |   8  |   9  |  10  |  11  |  12  |  13  |  14  |  15  |  16  |  17  |  18  |  19  |  20  |
|------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| Acc. | 66.6 | 77.2 | 78.6 | 83.2 | 88.2 | 85.8 | 82.4 | 85.4 | 87.6 | 90.8 | 91.0 | 84.6 | 89.2 | 83.0 | 56.2 | 75.4 | 71.0 | 73.8 | 90.6 | 93.6 |



---

## Experiment2 (Compact 6 fine-grained image tasks into ResNet50 network)  


**Step 1.** We provide the datasets of 5 tasks, including *cubs_cropped*, *stanford_cars_cropped*, *flowers*, *wikiart* and *sketches* (without *imagenet*), and they can be downloaded [here](https://drive.google.com/file/d/1a-FiCtYO_7nRcI9eIHrlZysq_0N3Sh2P/view?usp=sharing). After downloading, place the extracted directories in `data/`. If you would like to construct datasets yourself, please refer to [piggyback](https://github.com/arunmallya/piggyback). 


**Step 2.** Similar to Experiment1, we need to construct the accuracy goals for the 6 tasks. With the following command, we finetune the model pretrained on ImageNet to the other 5 tasks and produce accuracy goals stored in `logs/baseline_imagenet_acc_resnet50.txt`. 

```
$ bash experiment2/baseline_imagenet.sh
```

Or we can simply use the results of individual ResNet50 networks reported in the [piggyback paper](https://arxiv.org/pdf/1801.06519.pdf) as follows. 

```
{"imagenet": "0.7616", "cubs_cropped": "0.8283", "stanford_cars_cropped": "0.9183", "flowers": "0.9656", "wikiart": "0.7560", "sketches": "0.8078"}
```

**Step 3.** Run CPG and choose the desired pruning ratio for each of the 6 tasks.

```
$ bash experiment2/CPG_imagenet.sh 
```

We use the above command to run CPG for the task specified by the `TARGET_TASK_ID` varaiable in `experiment2/CPG_imagenet.sh`. 

For example, start from *imagenet* (specified by `TARGET_TASK_ID=1`), we run the above command and then select the gradually pruned model to proceed to learn the next task. 

More specifically, we check the `record.txt` in the checkpoint path, like `checkpoints/CPG/experiment2/resnet50/imagenet/gradual_prune/record.txt`, and find that there are 0.1, 0.2, 0.3, ... , 0.95 pruning ratios with their corresponding accuracies. We select the appropriate pruning ratio whose accuracy is higher than (or at least close to) the *imagenet*'s accuracy goal. Supposed that 0.4 is the best pruning ratio, we copy the corresponding checkpoint to `gradual_prune/` as below. 

```
In checkpoint/CPG/resnet50/imagenet/gradual_prune/

$ cp 0.4/checkpoint-46.pth.tar ./checkpoint-46.pth.tar

```

At last, we modify `TARGET_TASK_ID` to 2 and execute `experiment2/CPG_imagenet.sh` again so that CPG proceeds to learn the next task. 

We repeat **Step 3.** to sequentially learn the 6 tasks. 


**Step 4.** Inference the learned 6 tasks. 

```
$ bash experiment2/inference_CPG_imagenet.sh
```


### CPG-ResNet50 [Checkpoints](https://drive.google.com/file/d/1oYTQkNPo8JJ7lqKUKrAcu0T3ZAwe7C6r/view?usp=sharing) on Fine-grained Dataset.

Extract the downloaded .zip file and place `resnet50/` in `checkpoints/CPG/experiment2/`. 


| Task | ImageNet |  CUBS | Stanford Cars | Flowers | Wikiart | Sketch |
|:----:|:--------:|:-----:|:-------------:|:-------:|:-------:|:------:|
| Acc. |   75.81  | 83.57 |     92.81     |  96.55  |  76.98  |  80.33 |

---

## Experiment3 (Compact 4 facial-informatic tasks into CNN20 network)

**Step 1.** We provide the datasets of 3 tasks, including *emotion*, *gender* and *age* (without *face_verification*). For the *age* task, we adopt the 5-fold scenario and thus have *age0*, *age1*, ... , *age4* which correspond to the five splits. All face images are aligned using [MTCNN](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf) with output size of 112 x 112. The converted datasets can be downloaded [here](https://drive.google.com/file/d/1F2jx7k15EWA1P64Bp462ovB4zHb50tz_/view?usp=sharing). 


**Step 2.** Similarly, we need accuracy goals of the 4 tasks for CPG. We train CNN20 on VGGFace2 for the *face verification* task and finetune it to *emotion*, *gender* and *age* tasks. This [link](https://drive.google.com/file/d/1P3KiJGdanBbTpSFeLtbQCXIyoYrW8LtN/view?usp=sharing) provides our *face verification* CNN20 pretrained weights (named `face_weight.pth`), and the following command finetunes the model to other 3 tasks. To evaluate the *face verification* task, we also need `lfw_pairs.txt` which can be downloaded [here](https://drive.google.com/file/d/1wuKxHrDXebWicDxqt6FpBMhMbdN6wEDf/view?usp=sharing). Download `face_weight.pth` and `lfw_pairs.txt` use the links and place them in `face_data/`.

```
$ bash experiment3/baseline_face.sh 
```

The finetuning results are used as accuracy goals and stored in `logs/baseline_face_acc.txt`. You can also simply use the results as follows which corresponds to the finetuning results reported in our paper. 

```
{"face_verification": "0.9942", "gender": "0.9080", "emotion": "0.6254", "chalearn_gender": "0.9128", "age0": "0.6531", "age1": "0.5381", "age2": "0.5847", "age3": "0.5151", "age4": "0.5727"}
```

**Step 3.** Similar to Experiment2, we add tasks sequentially by iteratively running the following command and copy the pruned models with appropriate pruning ratios. 

```
$ bash experiment3/FvGeEm_CPG_face.sh 
```

Note that this script is only for learning the first 3 tasks, *face verification*, *gender* and *emotion*, by modifiying the `TARGET_TASK_ID` variable in it. Because we have 5 folds for the *age* task, use `experiment3/FvGeEmAg0_CPG_face.sh` for *age0*, `experiment3/FvGeEmAg1_CPG_face.sh` for *age1*, and so on. 

We repeat **Step 3.** until all 4 tasks, including 5 folds of the age task, are sequentially learned. 


**Step 4.** Inference the learned 4 tasks. 

```
$ bash experiment3/inference_FvGeEmAg.sh ${GPU_ID} ${TARGET_SPARSITY} ${AGEFOLD} ${LOG_PATH}
```

The inference script takes 4 arguments listed as follows: 
* GPU_ID: Index of GPU used to run the inference
* TARGET_SPARSITY: The target pruning ratio of the age task model to inference (like 0,1, 0.2, ..., 0.9, 0.95)
* AGEFOLD: The target fold of the 5 age folds to inference (like age0, age1, ..., age4)
* LOG_PATH: The target path to output inference log

For an example of using this script, please see `experiment3/inference_checkpoints_FvGeEmAg.sh`. 



### CPG-CNN20 [Checkpoints](https://drive.google.com/file/d/1TYcmaWm1kwkj4v-Zy43XuREyYOPIbghJ/view?usp=sharing) on Facial-informatic Dataset.

Extract the downloaded .zip file, place `spherenet20/` in `checkpoints/CPG/experiment3/` and use the following command for inference with the checkpoints. 

```
$ bash experiment3/inference_checkpoints_FvGeEmAg.sh 
```

| Task | Face verification |  Gender | Expression | Mean of 5-fold Age |
|:----:|:-----------------:|:-------:|:----------:|:------------------:|
| Acc. |  99.300 +- 0.348  |  89.66  |    63.57   |        57.66       |


---

## Benchmarking

### Cifar100 20 Tasks (datsets as experiment1 above) - VGG16 

| Methods |   1  |   2  |   3  |   4  |   5  |   6  |   7  |   8  |   9  |  10  |  11  |  12  |  13  |  14  |  15  |  16  |  17  |  18  |  19  |  20  | Avg. |
|:-------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| [PackNet](https://github.com/arunmallya/packnet) | 66.4 | 80.0 | 76.2 | 78.4 | 80.0 | 79.8 | 67.8 | 61.4 | 68.8 | 77.2 | 79.0 | 59.4 | 66.4 | 57.2 | 36.0 | 54.2 | 51.6 | 58.8 | 67.8 | 83.2 | 67.5 |
|   *[PAE](https://github.com/ivclab/PAE)   | 67.2 | 77.0 | 78.6 | 76.0 | 84.4 | 81.2 | 77.6 | 80.0 | 80.4 | 87.8 | 85.4 | 77.8 | 79.4 | 79.6 | 51.2 | 68.4 | 68.6 | 68.6 | 83.2 | 88.8 | 77.1 |
|   **CPG**   | 65.2 | 76.6 | 79.8 | 81.4 | 86.6 | 84.8 | 83.4 | 85.0 | 84.2 | 89.2 | 90.8 | 82.4 | 85.6 | 85.2 | 53.2 | 84.4 | 70.0 | 73.4 | 88.8 | 94.8 | 80.9 |

*PAE is our previous work.
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/compacting-picking-and-growing-for/continual-learning-on-cifar100-20-tasks)](https://paperswithcode.com/sota/continual-learning-on-cifar100-20-tasks?p=compacting-picking-and-growing-for)

### Fine-grained 6 Tasks (datsets as experiment2 above) - ResNet50

|       Methods      | ImageNet |  CUBS | Stanford Cars | Flowers | Wikiart | Sketch | Model Size (MB) |
|:------------------:|:--------:|:-----:|:-------------:|:-------:|:-------:|:------:|:---------------:|
| Train from Scratch |   76.16  | 40.96 |     61.56     |  59.73  |  56.50  |  75.40 |       554       |
|      Finetune      |     -    | 82.83 |     91.83     |  96.56  |  75.60  |  80.78 |       551       |
|   [ProgressiveNet](https://arxiv.org/abs/1606.04671)   |   76.16  | 78.94 |     89.21     |  93.41  |  74.94 |  76.35 |       563       |
|       [PackNet](https://github.com/arunmallya/packnet)      |   75.71  | 80.41 |     86.11     |  93.04  |  69.40  |  76.17 |       115       |
|      [Piggyback](https://github.com/arunmallya/piggyback)     |   76.16  | 84.59 |     89.62     |  94.77  |  71.33  |  79.91 |       121       |
|         **CPG**        |   75.81  | 83.59 |     92.80     |  96.62  |  77.15  |  80.33 |       121       |

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/compacting-picking-and-growing-for/continual-learning-on-imagenet-fine-grained-6)](https://paperswithcode.com/sota/continual-learning-on-imagenet-fine-grained-6?p=compacting-picking-and-growing-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/compacting-picking-and-growing-for/continual-learning-on-cubs-fine-grained-6)](https://paperswithcode.com/sota/continual-learning-on-cubs-fine-grained-6?p=compacting-picking-and-growing-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/compacting-picking-and-growing-for/continual-learning-on-stanford-cars-fine)](https://paperswithcode.com/sota/continual-learning-on-stanford-cars-fine?p=compacting-picking-and-growing-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/compacting-picking-and-growing-for/continual-learning-on-flowers-fine-grained-6)](https://paperswithcode.com/sota/continual-learning-on-flowers-fine-grained-6?p=compacting-picking-and-growing-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/compacting-picking-and-growing-for/continual-learning-on-wikiart-fine-grained-6)](https://paperswithcode.com/sota/continual-learning-on-wikiart-fine-grained-6?p=compacting-picking-and-growing-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/compacting-picking-and-growing-for/continual-learning-on-sketch-fine-grained-6)](https://paperswithcode.com/sota/continual-learning-on-sketch-fine-grained-6?p=compacting-picking-and-growing-for)

### Facial-informatic 4 Tasks (datasets as experiment3 above) - CNN20 

|       Methods      |      Face     |  Gender | Expression |  Age  |
|:------------------:|:-------------:|:-------:|:----------:|:-----:|
| Train from Scratch | 99.417+-0.367 |  83.70  |    57.64   | 46.14 | 
|      Finetune      |       -       |  90.80  |    62.54   | 57.27 | 
|      **CPG**       | 99.300+-0.348 |  89.66  |    63.57   | 57.66 | 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/compacting-picking-and-growing-for/age-and-gender-classification-on-adience-age)](https://paperswithcode.com/sota/age-and-gender-classification-on-adience-age?p=compacting-picking-and-growing-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/compacting-picking-and-growing-for/age-and-gender-classification-on-adience)](https://paperswithcode.com/sota/age-and-gender-classification-on-adience?p=compacting-picking-and-growing-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/compacting-picking-and-growing-for/facial-expression-recognition-on-affectnet)](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet?p=compacting-picking-and-growing-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/compacting-picking-and-growing-for/face-verification-on-labeled-faces-in-the)](https://paperswithcode.com/sota/face-verification-on-labeled-faces-in-the?p=compacting-picking-and-growing-for)

## Contact
Please feel free to leave suggestions or comments to [Steven C. Y. Hung](https://github.com/fevemania)(brent12052003@gmail.com), Cheng-Hao Tu(andytu455176@gmail.com), Cheng-En Wu(chengen@iis.sinica.edu.tw), [Chein-Hung Chen](https://github.com/Chien-Hung)(redsword26@gmail.com), [Yi-Ming Chan](https://github.com/yimingchan)(yiming@iis.sinica.edu.tw), [Chu-Song Chen](https://www.iis.sinica.edu.tw/pages/song/)(song@iis.sinica.edu.tw)
