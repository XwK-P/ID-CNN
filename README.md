# ID-CNN
SAR Image Despeckling Using a Convolutional Neural Network

## Train
```bash
DATA_ROOT=./dataset name=despeckle which_direction=BtoA th train.lua
```

## Test
```bash
DATA_ROOT=./dataset name=despeckle which_direction=BtoA phase=test th test.lua
```
