# ID-CNN
SAR Image Despeckling Using a Convolutional Neural Network

[[Paper Link](https://arxiv.org/abs/1706.00552)]  
## Train
```bash
DATA_ROOT=./dataset name=despeckle which_direction=BtoA th train.lua
```

## Test
```bash
DATA_ROOT=./dataset name=despeckle which_direction=BtoA phase=test th test.lua
```
