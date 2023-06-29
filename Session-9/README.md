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
