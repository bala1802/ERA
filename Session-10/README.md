# Objective

The purpose of this respository is to create Custom ResNet model for the CIFA10 dataset.

# How to read this repository?

- Clone the modular code to your notebook `git clone "https://github.com/bala1802/modular.git"`
- Install the required libraries from the modular/requirements.txt `pip install -r modular/requirements.txt`
- Execute the `Session-10.ipynb` notebook

# About Modular Code

- The Modular architecture is followed to keep the data processing activities and model build activities separately.
- `custom-resnet.py` This Script holds the CustomResNet model architecture
- `dataloader.py` This Script is to construct the loaders for the train and test datasets.
- `datautils.py` This Script is to perform data building activities.
- `modelutils.py` This Script is to perform model loading, hyperparameter finetuning, LRFinder, LRScheduler activities.
- `params.yaml` This yaml file is used to store and retrieve the parameters required for the fine-tuning
- `train.py` This Script is used to train the custom resnet model
- `test.py` This Script is used to test the custom resnet model

# Model Architecture

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4          [-1, 128, 34, 34]           8,192
       BatchNorm2d-5          [-1, 128, 34, 34]             256
              ReLU-6          [-1, 128, 34, 34]               0
         MaxPool2d-7          [-1, 128, 17, 17]               0
            Conv2d-8          [-1, 128, 17, 17]         147,456
       BatchNorm2d-9          [-1, 128, 17, 17]             256
             ReLU-10          [-1, 128, 17, 17]               0
           Conv2d-11          [-1, 128, 17, 17]         147,456
      BatchNorm2d-12          [-1, 128, 17, 17]             256
             ReLU-13          [-1, 128, 17, 17]               0
           Conv2d-14          [-1, 256, 19, 19]          32,768
      BatchNorm2d-15          [-1, 256, 19, 19]             512
             ReLU-16          [-1, 256, 19, 19]               0
        MaxPool2d-17            [-1, 256, 9, 9]               0
           Conv2d-18          [-1, 512, 11, 11]         131,072
      BatchNorm2d-19          [-1, 512, 11, 11]           1,024
             ReLU-20          [-1, 512, 11, 11]               0
        MaxPool2d-21            [-1, 512, 5, 5]               0
           Conv2d-22            [-1, 512, 5, 5]       2,359,296
      BatchNorm2d-23            [-1, 512, 5, 5]           1,024
             ReLU-24            [-1, 512, 5, 5]               0
           Conv2d-25            [-1, 512, 5, 5]       2,359,296
      BatchNorm2d-26            [-1, 512, 5, 5]           1,024
             ReLU-27            [-1, 512, 5, 5]               0
AdaptiveAvgPool2d-28            [-1, 512, 1, 1]               0
           Linear-29                   [-1, 10]           5,130
----------------------------------------------------------------
- Total params: 5,196,874
- Trainable params: 5,196,874
- Non-trainable params: 0
----------------------------------------------------------------
- Input size (MB): 0.01
- Forward/backward pass size (MB): 11.24
- Params size (MB): 19.82
- Estimated Total Size (MB): 31.08
----------------------------------------------------------------


