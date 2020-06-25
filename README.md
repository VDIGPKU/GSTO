# GSTO: Gated Scale-Transfer Operation for Multi-Scale Feature Learning in Pixel Labeling
This is the official implementation of paper: GSTO: Gated Scale-Transfer Operation for Multi-Scale Feature Learning in Pixel Labeling

by Zhuoying Wang, Yongtao Wang.

## Introduction

Contact us with wzypku@pku.edu.cn, wyt@pku.edu.cn.

**The project is only free for academic research purposes, but needs authorization for commerce. For commerce permission, please contact wyt@pku.edu.cn.**


## Citation

If you use our code/model/data, please cite our paper:
https://arxiv.org/abs/2005.13363



## Semantic Segmentation Results

### Cityscapes val set

single scale and no flipping, without using OHEM

|     model    | #Params |  GFLOPs | mIoU |                 
|:------------:|:-------:|:-------:|:----:|
|GSTO-HRNet-W48|  65.93M |  714.0  | 82.1 |

### Cityscapes test set

multi-scale with flipping

|     model    | Train set |   OHEM  | mIoU |                 
|:------------:| :-------: |:-------:|:----:|
|GSTO-HRNet-W48|   Train   |  False  | 81.9 |
|GSTO-HRNet-W48|  Trainval |  False  | 82.3 |
|GSTO-HRNet-W48|  Trainval |  True   | 82.4 |

### LIP 
|     model    |   Extra.  | Pixel acc. | avg acc. | mIoU |              
|:------------:| :-------: | :-------:  | :------: |:----:|
|GSTO-HRNet-W48|   Train   |   88.38    |  68.36   | 57.4 |

## Pose Estimation Results

### COCO val set
|     model    |Input size | #Params |  GFLOPs |  AP  |                
|:------------:| :-------: |:-------:|:-------:|:----:|
|GSTO-HRNet-W32|  384x288  |  29.6M  |   18.2  | 76.5 |
|GSTO-HRNet-W48|  384x288  |  66.0M  |   37.6  | 76.7 |
 
 
### COCO test set
|     model    |Input size | #Params |  GFLOPs |  AP  |                
|:------------:| :-------: |:-------:|:-------:|:----:|
|GSTO-HRNet-W32|  384x288  |  29.6M  |   18.2  | 75.5 |
|GSTO-HRNet-W48|  384x288  |  66.0M  |   37.6  | 75.8 |

## Quick inference
### Install
1. PyTorch=0.4.1 
2. git clone https://github.com/VDIGPKU/GSTO
3. install dependencies 
    EasyDict==1.7
    opencv-python==3.4.1.15
    shapely==1.6.4
    Cython
    scipy
    pandas
    pyyaml
    json_tricks
    scikit-image
    yacs>=0.1.5
    tensorboardX>=1.6
    tqdm
    ninja
    
### Data 
follow the prepare instruction in [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation#data-preparation)

### Test

#### cityscapes test result 82.4
```Bash
CUDA_VISIBLE_DEVICES=0 python test.py --cfg cfg_files/gsto_hrnet_w48_cityscapes.yaml
```

#### LIP test result 
```Bash
CUDA_VISIBLE_DEVICES=0 python test.py --cfg cfg_files/gsto_hrnet_w48_lip.yaml
```
