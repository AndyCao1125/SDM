# ANN2SNN Spiking DDPM

This repo holds the codes for Spiking Denoising Diffusion Probabilistic Model through quantized ANN-SNN conversion.

This implementation realizes a spiking Unet model based on the work from Fast-SNN [1].

## Requirements

- Python 3.9.12

- Install all the dependencies
  
  ```
  pip install -r requirements.txt
  ```

- Download dataset:
   For cifar10 dataset, create folder `stats` for `cifar10.train.npz`.
   (download link:[cifar10.train.npz](https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC?usp=sharing))

## Prepare Quantized ANNs

We use APoT method to train our quantized ANNs. [APoT_Quantization](https://github.com/yhhhli/APoT_Quantization)

### CIFAR-10

The whole process contains:

- quantized ANNs training
- SNNs conversion
- SNNs fine-tune
- SNNs evaluation.

You can specify your implementation details of these four process by modifying the flagfiles which are contained in folders `./config/train_ANN/CIFAR10.txt`, `./config/SNN_convert/CIFAR10_snn.txt`, `./config/SNN_ft/CIFAR10_snnft.txt` and `./config/SNN_evaluate/CIFAR10_snn.txt`. And you should set the same activation precision `--bit` for these `.txt` files.

Note that the time steps of SNNs are automatically calculated from activation precision, i.e., T = 2^b-1.

#### Train Quantized ANNs

Examples to train Quantized ANNs.

- [example 1]: Overwrite arguments and select GPU IDs
  
  ```
  CUDA_VISIBLE_DEVICES=0 python ANN_train.py \
      --flagfile ./config/train_ANN/CIFAR10.txt
  ```

- [example 2]: Use multi-GPU for training
  
  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3 python ANN_train.py \
      --flagfile ./config/train_ANN/CIFAR10.txt \
      --parallel
  ```
  
  Note that once you specify the activation precision and start training for the first time, a corresponding folder will be set up in folder `logs`. You can give a name to this folder in the flagfile by setting `--logdir`. It will contain model weight files and results of following experiments.

#### Convert Quantized ANNs to SNNs

Load and convert the pretrained quantized ANNs to SNNs. 
Examples to implement ANN-SNN conversion without fine-tune.

- [example 1]: Overwrite arguments, select GPU IDs.
  
  ```
  CUDA_VISIBLE_DEVICES=0 python SNN_conversion.py \
      --flagfile ./config/SNN_convert/CIFAR10_snn.txt
  ```

- [example 2]: Use multi-GPU.
  
  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3 python SNN_conversion.py \
      --flagfile ./config/SNN_convert/CIFAR10_snn.txt \
      --parallel
  ```

The images generated are in the `images` folder. The weight file name of converted SNNs is `converted_snn.pt`

#### Fine-tune Converted SNNs

Load, convert and fine-tune the pretrained quantized ANNs to SNNs. We use a **block-wise** fine-tune method.
Examples to implement ANN-SNN conversion with fine-tune.

- [example 1]: Overwrite arguments, select GPU IDs.
  
  ```
  CUDA_VISIBLE_DEVICES=0 python SNN_ft.py \
      --flagfile ./config/SNN_ft/CIFAR10_snnft.txt
  ```

- [example 2]: Use multi-GPU.
  
  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3 python SNN_ft.py \
      --flagfile ./config/SNN_ft/CIFAR10_snnft.txt \
      --parallel
  ```

The results and weight files of the fine-tune operation are included in folder with the prefix `SNN_` in the logdir file.

example:

```
logs
 └── RELU_QUANT2B_DDPM_CIFAR10_EPS
                └── SNN_2.0bit_ft
                          └── ft_blocks_samples
                          └── model_backpacks
```



#### Evaluate SNNs

Load and evaluate converted SNNs with following examples. Set the path to the weight file that you want to evaluate in the flagfile `--eval_model_path` first  

- [example 1]: Overwrite arguments, select GPU IDs.
  
  ```
  CUDA_VISIBLE_DEVICES=0 python evaluate.py \
      --flagfile ./config/SNN_evaluate/CIFAR10_snn.txt
  ```

- [example 2]: Use multi-GPU.
  
  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate.py \
      --flagfile ./config/SNN_evaluate/CIFAR10_snn.txt \
      --parallel
  ```

The results and images will be saved in `snn_eval.txt` and `images`.

#### FID and IS of Spiking DDPM through ANN2SNN for cifar10

| Activation precision | Fine-tune | FID     | IS     |
|:--------------------:|:---------:|:-------:|:------:|
| 2 bit                | Yes       | 29.5350 | 7.2232 |
| 2 bit                | No        | 51.1811 | 5.4837 |

## Reference

[1] [Fast-SNN](https://github.com/yangfan-hu/Fast-SNN)
