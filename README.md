## Progressive Image Deraining Networks: PRN_dense & PRN_dense_conv

### Introduction

Typical de-raining methods involve decomposing the image into a background layer and a rain layer and attempting to separate these layers. However, this can be difficult, especially in complex weather conditions, and if the algorithm over- or underestimates the amount of rain in the image, the resulting image may be fuzzy or still contain fragmented raindrops. To conduct feature-guided image de-raining and minimize the negative impact of rain removal on image quality, it is proposed to replace residual connection with the dense connection in PReNet model.
Two models **_PRN_dense_** and **_PRN_dense_conv_** are presented.

## Prerequisites

- Python 3.9, PyTorch 2.0.0 + cu117
- Requirements: opencv-python, tensorboardX
- Platforms: Windows 10, cuda-12.1.0
- MATLAB

## Datasets

PRN and PReNet are evaluated on two datasets:
Rain100H and Rain100L.
Please download the testing and training datasets from [Google Drive](https://drive.google.com/file/d/15EpJFX0K6tNdIjpoacmjHU4LUt8Mmfqf/view?usp=sharing),
and place the unzipped folders into `./`.

### 1) Training

Run shell scripts to train the models:

```bash
bat train_PRN_dense.bat
bat train_PRN_dense_conv.bat
```

You can use `tensorboard --logdir ./logs/your_model_path` to check the training procedures.

### 2) Testing

The pre-trained models are in the path `./logs/`.

Run bat scripts to test the models:

```bash
test_Rain100H.bat   # test models on Rain100H
test_Rain100L.bat   # test models on Rain100L
test_real.bay       # test models on real rainy images
```

### 3) Evaluation metrics

MATLAB scripts are provided to compute the average PSNR and SSIM values reported in the paper.

```Matlab
 cd ./statistic
 run statistic_Rain100H.m
 run statistic_Rain100L.m
```

## References
 D. Ren, W. Zuo, Q. Hu, P. Zhu, and D. Meng, “Progressive Image Deraining Networks: A Better and Simpler Baseline,” 2019, doi: 10.48550/arxiv.1901.09221.

 https://github.com/csdwren/PReNet
