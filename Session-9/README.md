# Objective

Construct a Neural Network for the CIFAR10 dataset Multiclass classification, based on the below constraints

- The architecture should have 4 different blocks with 1 output block.
- Each block should have only 3 convolution layer.
- One of the layers must be Depthwise Seperable Convolution
- One of the layers must use Dilated Convolution
- Complusory Global Average Pooling post that Fully Connected Layer should be added
- The total Receptive fields used must be more than 44
- The data transformations should be done using `albumentations` library (`pip install albumentations`)
- Albumentations specifications:
  - Horizontal Flip
  - Shift Scale Rotate
  - Cutout
 
# How to read this repository?

The Session09.ipynb is the main notebook, inside which the `model.py` and `util.py` are used as helper class.

# `model.py`

This script is used to construct the Neural Network with below specifications

Architecture:

    - C1, C2, C3, C4
    - Each block will have 3 Convolution layers.
    - First layer will be Depthwise Seperable convolution layer; 
    - Second Layer will be a 1X1
    - Third layer will be a convolution layer with dilation as 2

Model Summary:

----------------------------------------------------------------

        Layer (type)               Output Shape         Param #
        
----------------------------------------------------------------

            Conv2d-1           [-1, 60, 32, 32]             540
       BatchNorm2d-2           [-1, 60, 32, 32]             120
              ReLU-3           [-1, 60, 32, 32]               0
            Conv2d-4           [-1, 30, 34, 34]           1,800
       BatchNorm2d-5           [-1, 30, 34, 34]              60
              ReLU-6           [-1, 30, 34, 34]               0
            Conv2d-7           [-1, 60, 30, 30]          16,200
       BatchNorm2d-8           [-1, 60, 30, 30]             120
              ReLU-9           [-1, 60, 30, 30]               0
           Conv2d-10          [-1, 120, 30, 30]           1,080
      BatchNorm2d-11          [-1, 120, 30, 30]             240
             ReLU-12          [-1, 120, 30, 30]               0
           Conv2d-13           [-1, 30, 32, 32]           3,600
      BatchNorm2d-14           [-1, 30, 32, 32]              60
             ReLU-15           [-1, 30, 32, 32]               0
           Conv2d-16           [-1, 60, 28, 28]          16,200
      BatchNorm2d-17           [-1, 60, 28, 28]             120
             ReLU-18           [-1, 60, 28, 28]               0
           Conv2d-19          [-1, 120, 28, 28]           1,080
      BatchNorm2d-20          [-1, 120, 28, 28]             240
             ReLU-21          [-1, 120, 28, 28]               0
           Conv2d-22           [-1, 30, 28, 28]           3,600
      BatchNorm2d-23           [-1, 30, 28, 28]              60
             ReLU-24           [-1, 30, 28, 28]               0
           Conv2d-25           [-1, 60, 24, 24]          16,200
      BatchNorm2d-26           [-1, 60, 24, 24]             120
             ReLU-27           [-1, 60, 24, 24]               0
           Conv2d-28          [-1, 120, 22, 22]           1,080
      BatchNorm2d-29          [-1, 120, 22, 22]             240
             ReLU-30          [-1, 120, 22, 22]               0
           Conv2d-31           [-1, 30, 22, 22]           3,600
      BatchNorm2d-32           [-1, 30, 22, 22]              60
             ReLU-33           [-1, 30, 22, 22]               0
           Conv2d-34           [-1, 60, 18, 18]          16,200
      BatchNorm2d-35           [-1, 60, 18, 18]             120
             ReLU-36           [-1, 60, 18, 18]               0
        AvgPool2d-37             [-1, 60, 1, 1]               0
           Conv2d-38             [-1, 10, 1, 1]             600
           
----------------------------------------------------------------

- Total params: 83,340
- Trainable params: 83,340
- Non-trainable params: 0

----------------------------------------------------------------

- Input size (MB): 0.01
- Forward/backward pass size (MB): 13.28
- Params size (MB): 0.32
- Estimated Total Size (MB): 13.61

----------------------------------------------------------------

# Albumentations

- Albumentations is a popular library for image augmentation in machine learning and computer vision tasks.
- It offers a wide range of efficient augmentation techniques for tasks like object detection and image classification.
- Albumentations is fast, memory-efficient, and supports various input image formats.
- It allows customizable augmentation pipelines and can generate augmented images on the fly during training.
- It is particularly useful for reducing storage requirements and handling dynamic datasets.

## Albumentations techniques applied in this repository:

- `A.HorizontalFlip(p=0.5)`: flips an image horizontally with a 50% probability.
- `A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5)`: applies random shifts, scales, and rotations to an image with specified limits and a 50% probability.
- `A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16, fill_value=[0.4914, 0.4822, 0.4465], mask_fill_value = None, p=0.5),`: applies coarse dropout by randomly removing square-shaped regions from an image, with specified maximum and minimum hole sizes, fill values, and a 50% probability.
