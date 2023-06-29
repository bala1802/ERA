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
