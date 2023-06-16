import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Model-1:

- 2 Max Pooling layers are added
- The Kernel size is 3 across the layers
- The number of channels is increased continuosly across the layers, except the 7th Convolutional Layer, 
where the number of channels is reduced from 1024 to 10 (10 is the number of target labels)
- The activation function relu is applied across all the layers except the Fully Connected layer

Input| Output Channels | Layers         | Receptive field                   Shape (rename this)
-----------------------------------------------------------------------------------------------
1    | 32     Convolutional layer-1     | 3  -> 1st 3X3 kernel convolved  | 28X28
32   | 64     Convolutional layer-2     | 5  -> 2nd 3X3 kernel convolved  | 28X28
Max Pooling Layer-1                     | 10 -> 1st MaxPooling            | 14X14
64   | 128    Convolutional layer-3     | 12 -> 3rd 3X3 kernel convolved  | 14X14
128  | 256    Convolutional layer-4     | 14 -> 4th 3X3 kernel convolved  | 14X14
Max Pooling Layer-2                     | 28 -> 2nd MaxPooling            | 7X7
256  | 512    Convolutional layer-5     | 30 -> 5th 3X3 kernel convolved  | 5X5
512  | 1024   Convolutional layer-6     | 32 -> 6th 3X3 kernel convolved  | 3X3
1024 | 10     Convolutional layer-7     | 34 -> 7th 3X3 kernel convolved  | 1X1

--------------------------------------------------------------------------------------------------------------------------------------------------------
        Layer (type)               Output Shape         Param #   (input_channels * output_channels * kernel_height * kernel_width) + output_channels
========================================================================================================================================================
            Conv2d-1           [-1, 32, 28, 28]             320   (1  *  32  * 3 * 3) + 32 = 320
            Conv2d-2           [-1, 64, 28, 28]          18,496   (32 *  64  * 3 * 3) + 64 = 18,496
         MaxPool2d-3           [-1, 64, 14, 14]               0   (No parameters in max pooling)
            Conv2d-4          [-1, 128, 14, 14]          73,856   (64 *  128 * 3 * 3) + 128 = 73,856
            Conv2d-5          [-1, 256, 14, 14]         295,168   (128 * 256 * 3 * 3) + 256 = 295,168
         MaxPool2d-6            [-1, 256, 7, 7]               0   (No parameters in max pooling)
            Conv2d-7            [-1, 512, 5, 5]       1,180,160   (256 * 512 * 3 * 3) + 512 = 1,180,160
            Conv2d-8           [-1, 1024, 3, 3]       4,719,616   (512 * 1024 * 3 * 3) + 1024 = 4,719,616
            Conv2d-9             [-1, 10, 1, 1]          92,170   (1024 * 10 * 3 * 3) + 10 = 92,170
========================================================================================================================================================
Total params: 6,379,786
Trainable params: 6,379,786
Non-trainable params: 0
--------------------------------------------------------------------------------------------------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.51
Params size (MB): 24.34
Estimated Total Size (MB): 25.85
--------------------------------------------------------------------------------------------------------------------------------------------------------
'''

class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # 28>28 | 3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 28 > 28 |  5
        self.pool1 = nn.MaxPool2d(2, 2) # 28 > 14 | 10
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 14> 14 | 12
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) #14 > 14 | 14
        self.pool2 = nn.MaxPool2d(2, 2) # 14 > 7 | 28
        self.conv5 = nn.Conv2d(256, 512, 3) # 7 > 5 | 30
        self.conv6 = nn.Conv2d(512, 1024, 3) # 5 > 3 | 32 | 3*3*1024 | 3x3x1024x10 | 
        self.conv7 = nn.Conv2d(1024, 10, 3) # 3 > 1 | 34 | > 1x1x10

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        # x = F.relu(self.conv7(x))
        x = self.conv7(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)

'''
Model-2

- In the 3rd block, the channels are reduced from 256->128->64
- Post that it is reduced to 10

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 28, 28]             320
            Conv2d-2           [-1, 64, 28, 28]          18,496
         MaxPool2d-3           [-1, 64, 14, 14]               0
            Conv2d-4          [-1, 128, 14, 14]          73,856
            Conv2d-5          [-1, 256, 14, 14]         295,168
         MaxPool2d-6            [-1, 256, 7, 7]               0
            Conv2d-7            [-1, 128, 5, 5]         295,040
            Conv2d-8             [-1, 64, 3, 3]          73,792
            Conv2d-9             [-1, 10, 1, 1]           5,770
================================================================
Total params: 762,442
Trainable params: 762,442
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.37
Params size (MB): 2.91
Estimated Total Size (MB): 4.28
----------------------------------------------------------------
'''
class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # 28 > 28 | 3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 28 > 28 | 5
        self.pool1 = nn.MaxPool2d(2, 2) # 28 > 28 | 10
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 14 > 14 | 12
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) # 14 > 14 | 14
        self.pool2 = nn.MaxPool2d(2, 2) # 14 > 7 | 28
        self.conv5 = nn.Conv2d(256, 128, 3) # 7 > 5 | 30
        self.conv6 = nn.Conv2d(128, 64, 3) # 5 > 3 | 32
        self.conv7 = nn.Conv2d(64, 10, 3) # 3 > 1 | 34
        
    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = self.conv7(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)
         
'''
Model-3

- In the end layers, the channels are reduced from 64->32->16
- Then the channels are converted to the number of labels 10
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 4, 28, 28]              40
            Conv2d-2           [-1, 16, 28, 28]             592
         MaxPool2d-3           [-1, 16, 14, 14]               0
            Conv2d-4           [-1, 32, 14, 14]           4,640
            Conv2d-5           [-1, 64, 14, 14]          18,496
         MaxPool2d-6             [-1, 64, 7, 7]               0
            Conv2d-7             [-1, 32, 5, 5]          18,464
            Conv2d-8             [-1, 16, 3, 3]           4,624
            Conv2d-9             [-1, 10, 1, 1]           1,450
================================================================
Total params: 48,306
Trainable params: 48,306
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.32
Params size (MB): 0.18
Estimated Total Size (MB): 0.51
----------------------------------------------------------------
'''
class Model_3(nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv5 = nn.Conv2d(64, 32, 3)
        self.conv6 = nn.Conv2d(32, 16, 3)
        self.conv7 = nn.Conv2d(16, 10, 3)
    
    def forward(self, x):
        x = F.relu(self.conv2(F.relu(self.conv1(x))))
        x = self.pool1(x)
        x = F.relu(self.conv4(F.relu(self.conv3(x))))
        x = self.pool2(x)
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

'''
Model-4

- The paddings are removed from the initial layers and added in the middle convolutional layers
- In total 3 max pooling layers are added
- the 3rd maxpooling layer is added just before the prediction layer
'''
class Model_4(nn.Module):
    def __init__(self):
        super(Model_4, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.conv2 = nn.Conv2d(4, 16, 3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv5 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 16, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv7 = nn.Conv2d(16, 10, 3)
        
    def forward(self, x):
        x = F.relu(self.conv2(F.relu(self.conv1(x))))
        x = self.pool1(x)
        x = F.relu(self.conv4(F.relu(self.conv3(x))))
        x = self.pool2(x)
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = self.pool3(x)
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

'''
Model-5

Let's build a lighter model, by having a drastic decrement in the number of parameters
'''
class Model_5(nn.Module):
    def __init__(self):
        super(Model_5, self).__init__()
        #Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
            ) #output size = 26
        
        #Convolutional Block-1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3,3), padding=0, bias=False),
            nn.ReLU()
            ) #output size = 24
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3,3), padding=0, bias=False),
            nn.ReLU()
            ) #output size = 22
        
        #Transitional Block-1
        self.pool1 = nn.MaxPool2d(2,2) #output size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU()
            ) #output size = 11
        
        #Convolutional Block-2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3,3), padding=0, bias=False),
            nn.ReLU()
            ) #output size = 9
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3,3), padding=0, bias=False),
            nn.ReLU()
            ) #output size = 7
        
        #Output Block
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.ReLU()
            ) # output size = 7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7,7), padding=0, bias=False)
            #Never apply Batch Normalization and the activation function in the last layer of the Neural Network
            )
    
    def forward(self, x):
        #print("Phase1 : ", x.shape) # torch.Size([2, 1, 28, 28])
        x = self.convblock1(x)
        #print("Phase2 : ", x.shape) # torch.Size([2, 10, 26, 26])
        x = self.convblock2(x)
        #print("Phase3 : ", x.shape) # torch.Size([2, 10, 24, 24])
        x = self.convblock3(x)
        #print("Phase4 : ", x.shape) # torch.Size([2, 20, 22, 22])
        x = self.pool1(x)
        #print("Phase5 : ", x.shape) # torch.Size([2, 20, 11, 11])
        x = self.convblock4(x)
        #print("Phase6 : ", x.shape) # torch.Size([2, 10, 11, 11])
        x = self.convblock5(x)
        #print("Phase7 : ", x.shape) # torch.Size([2, 10, 9, 9])
        x = self.convblock6(x)
        #print("Phase8 : ", x.shape) # torch.Size([2, 20, 7, 7])
        x = self.convblock7(x)
        #print("Phase9 : ", x.shape) # torch.Size([2, 10, 7, 7])
        x = self.convblock8(x)
        #print("Phase10 : ", x.shape) # torch.Size([2, 10, 1, 1])
        x = x.view(-1, 10)
        #print("Phase11 : ", x.shape) # torch.Size([2, 10])
        softmax = F.log_softmax(x, dim=-1) 
        #print("Softmax : ", softmax)
        #print("Softmax-Shape : ", softmax.shape) # torch.Size([2, 10])
        #print("------------------------------------------------------------")
        return softmax
        
        
'''
Model-6

- Added Batch Normalization to all the Convolutional Layers

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
       BatchNorm2d-2           [-1, 10, 26, 26]              20
              ReLU-3           [-1, 10, 26, 26]               0
            Conv2d-4           [-1, 10, 24, 24]             900
       BatchNorm2d-5           [-1, 10, 24, 24]              20
              ReLU-6           [-1, 10, 24, 24]               0
            Conv2d-7           [-1, 20, 22, 22]           1,800
       BatchNorm2d-8           [-1, 20, 22, 22]              40
              ReLU-9           [-1, 20, 22, 22]               0
        MaxPool2d-10           [-1, 20, 11, 11]               0
           Conv2d-11           [-1, 10, 11, 11]             200
      BatchNorm2d-12           [-1, 10, 11, 11]              20
             ReLU-13           [-1, 10, 11, 11]               0
           Conv2d-14             [-1, 10, 9, 9]             900
      BatchNorm2d-15             [-1, 10, 9, 9]              20
             ReLU-16             [-1, 10, 9, 9]               0
           Conv2d-17             [-1, 20, 7, 7]           1,800
      BatchNorm2d-18             [-1, 20, 7, 7]              40
             ReLU-19             [-1, 20, 7, 7]               0
           Conv2d-20             [-1, 10, 7, 7]             200
      BatchNorm2d-21             [-1, 10, 7, 7]              20
             ReLU-22             [-1, 10, 7, 7]               0
           Conv2d-23             [-1, 10, 1, 1]           4,900
================================================================
Total params: 10,970
Trainable params: 10,970
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.61
Params size (MB): 0.04
Estimated Total Size (MB): 0.65
----------------------------------------------------------------
'''
class Model_6(nn.Module):
    def __init__(self):
        super(Model_6, self).__init__()
        #Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3,3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
            ) #output size = 26
        
        #Convolutional Block-1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3,3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
            ) #output size = 24
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3,3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
            ) #output size = 22
        
        #Transition Block-1
        self.pool1 = nn.MaxPool2d(2,2) #output size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
            ) #output size = 11
        
        #Convolution Block-2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3,3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
            ) #output size = 9
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3,3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
            ) #output size = 7
        
        #Output Block
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
            ) #output size = 7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7, 7), padding=0, bias=False),
            #Never apply Batch Normalization and activation function in the last layers
            ) #output size = 1
    
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

'''
Model-7

 - Adding the dropouts.
 - Observation needed on change in the number of parameters.
'''
class Model_7(nn.Module):
    def __init__(self):
        super(Model_7, self).__init__()
        #Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3,3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            
            nn.ReLU()
            ) #output size = 26
        
        #Convolutional Block-1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3,3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
            ) #output size = 24
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3,3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
            ) #output size = 22
        
        #Transition Block-1
        self.pool1 = nn.MaxPool2d(2,2) #output size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
            ) #output size = 11
        
        #Convolution Block-2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3,3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
            ) #output size = 9
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3,3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
            ) #output size = 7
        
        #Output Block
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
            ) #output size = 7
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7, 7), padding=0, bias=False),
            #Never apply Batch Normalization and activation function in the last layers
            ) #output size = 1
        
        #Regularization - dropout
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.dropout(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        
'''
Model-8

Applying Global Average Pooling

'''
class Model_8(nn.Module):
    def __init__(self):
        super(Model_8, self).__init__()
        # Input Block 28  >>> 64
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 26 >>> 62

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 24. >>> 60
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) # output_size = 22 >>> 58

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 11 >>> 29

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 9 >>> 27
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) # output_size = 7 >>> 25

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 7 >>> 25
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7) # 7>> 9... nn.AdaptiveAvgPool((1, 1))
        ) # output_size = 1

        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.dropout(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        
'''
Model-9
'''
class Model_9(nn.Module):
    def __init__(self):
        super(Model_9, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 24
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) # output_size = 22

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 11

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 9
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU()
        ) # output_size = 7

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 5
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 5
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        ) # output_size = 1

        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.dropout(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.dropout(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

'''
Model-10
'''
class Model_10(nn.Module):
    def __init__(self):
        super(Model_10, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)
        ) # output_size = 26, receptive_field = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.1)
        ) # output_size = 24, receptive_field = 5

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24, receptive_field = 5
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12, receptive_field = 10

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(14),
            nn.Dropout(0.1)
        ) # output_size = 10, receptive_field = 12
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        ) # output_size = 8, receptive_field = 14
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)
        ) # output_size = 6, receptive_field = 16
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(0.1)
        ) # output_size = 6, receptive_field = 18
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1, receptive_field = 24

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) #receptive_field = 24


        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

'''
Model-11
'''
class Model_11(nn.Module):
    def __init__(self):
        super(Model_11, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)
        ) # output_size = 26, receptive_field = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(0.1)
        ) # output_size = 24, receptive_field = 5

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24, receptive_field = 5
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12, receptive_field = 10

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(14),
            nn.Dropout(0.1)
        ) # output_size = 10, receptive_field = 12
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        ) # output_size = 8, receptive_field = 14
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(8),
            nn.Dropout(0.1)
        ) # output_size = 6, receptive_field = 16
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(0.1)
        ) # output_size = 6, receptive_field = 18
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1, receptive_field = 24

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) #receptive_field = 24


        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)