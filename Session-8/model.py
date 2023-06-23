import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Working Skeleton
'''
class Model_WS(nn.Module):
    def __init__(self):
        super(Model_WS, self).__init__()
        '''
        Convolutional BLOCK-1
        '''
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.GroupNorm(8, 32), #Dividing the 32 channels into 8 groups
            nn.ReLU()
            ) #output size = 32
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.GroupNorm(8, 32), #Dividing the 64 channels into 8 groups
            nn.ReLU()
            ) #output size = 32
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.GroupNorm(2, 10), #Dividing the 10 channels into 2 groups
            nn.ReLU()
            ) #output size = 32
        '''
        Transition BLOCK-1
        '''
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 16
        
        '''
        Convolutional BLOCK-2
        '''
        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.GroupNorm(8, 32), #Dividing the 64 channels into 8 groups
            nn.ReLU()
            ) #output size = 16
        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.GroupNorm(8, 32), #Dividing the 32 channels into 8 groups
            nn.ReLU()
            ) #output size = 16
        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.GroupNorm(8, 32), #Dividing the 32 channels into 8 groups
            nn.ReLU()
            ) #output size = 16
        self.c6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.GroupNorm(2, 10), #Dividing the 10 channels into 2 groups
            nn.ReLU()
            ) #output size = 16
        '''
        Transition BLOCK-2
        '''
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 8
        
        '''
        Convolutional BLOCK-3
        '''
        self.C7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=64, kernel_size=(3,3), padding=0, bias=False),
            nn.GroupNorm(8,64), #Dividing the 64 channels into 8 groups
            nn.ReLU()
            ) #output size = 6
        self.C8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=0, bias=False),
            nn.GroupNorm(8,64), #Dividing the 64 channels into 8 groups
            nn.ReLU()
            ) #output size = 4
        self.c9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1,1), padding=0, bias=False),
            nn.GroupNorm(2,32), #Dividing the 10 channels into 2 groups
            nn.ReLU()
            ) #output size = 4
        
        '''
        Global Average Pooling
        '''
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output size = 1
        
        self.C10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.c3(x)
        x = self.pool1(x)
        
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        x = self.c6(x)
        x = self.pool2(x)
        
        x = self.C7(x)
        x = self.C8(x)
        x = self.c9(x)
        x = self.gap(x)
        
        x = self.C10(x)
        
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)



'''
Model-1 - Neural Network with Group Normalization
'''

'''
Model-1 with Group Normalization
'''
class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        '''
        Convolutional BLOCK-1
        '''
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.GroupNorm(8, 32), #Dividing the 32 channels into 8 groups
            nn.ReLU()
            ) #output size = 32
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.GroupNorm(8, 32), #Dividing the 64 channels into 8 groups
            nn.ReLU()
            ) #output size = 32
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.GroupNorm(2, 10), #Dividing the 10 channels into 2 groups
            nn.ReLU()
            ) #output size = 32
        '''
        Transition BLOCK-1
        '''
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 16
        
        '''
        Convolutional BLOCK-2
        '''
        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.GroupNorm(8, 32), #Dividing the 64 channels into 8 groups
            nn.ReLU()
            ) #output size = 16
        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.GroupNorm(8, 32), #Dividing the 32 channels into 8 groups
            nn.ReLU()
            ) #output size = 16
        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.GroupNorm(8, 32), #Dividing the 32 channels into 8 groups
            nn.ReLU()
            ) #output size = 16
        self.c6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.GroupNorm(2, 10), #Dividing the 10 channels into 2 groups
            nn.ReLU()
            ) #output size = 16
        '''
        Transition BLOCK-2
        '''
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 8
        
        '''
        Convolutional BLOCK-3
        '''
        self.C7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(3,3), padding=0, bias=False),
            nn.GroupNorm(8,32), #Dividing the 32 channels into 8 groups
            nn.ReLU()
            ) #output size = 6
        self.C8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=0, bias=False),
            nn.GroupNorm(8,32), #Dividing the 32 channels into 8 groups
            nn.ReLU()
            ) #output size = 4
        self.c9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1), padding=0, bias=False),
            nn.GroupNorm(2,32), #Dividing the 32 channels into 2 groups
            nn.ReLU()
            ) #output size = 4
        
        '''
        Global Average Pooling
        '''
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output size = 1
        
        self.C10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            )
    
    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.c3(x)
        x = self.pool1(x)
        
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        x = self.c6(x)
        x = self.pool2(x)
        
        x = self.C7(x)
        x = self.C8(x)
        x = self.c9(x)
        x = self.gap(x)
        
        x = self.C10(x)
        
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        
'''
Model-2

Applied Layer Normalization
'''

class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        '''
        Convolutional BLOCK-1
        '''
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.GroupNorm(1, 32), #Dividing the 32 channels into 1 group - Layer Normalization
            nn.ReLU()
            ) #output size = 32
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.GroupNorm(1, 32), #Dividing the 32 channels into 1 group - Layer Normalization
            nn.ReLU()
            ) #output size = 32
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.GroupNorm(1, 10), #Dividing the 10 channels into 1 group - Layer Normalization
            nn.ReLU()
            ) #output size = 32
        '''
        Transition BLOCK-1
        '''
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 16
        
        '''
        Convolutional BLOCK-2
        '''
        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.GroupNorm(1, 32), #Dividing the 32 channels into 1 group - Layer Normalization
            nn.ReLU()
            ) #output size = 16
        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.GroupNorm(1, 32), #Dividing the 32 channels into 1 group - Layer Normalization
            nn.ReLU()
            ) #output size = 16
        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.GroupNorm(1, 32), #Dividing the 32 channels into 1 group - Layer Normalization
            nn.ReLU()
            ) #output size = 16
        self.c6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.GroupNorm(1, 10), #Dividing the 10 channels into 1 group - Layer Normalization
            nn.ReLU()
            ) #output size = 16
        '''
        Transition BLOCK-2
        '''
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 8
        
        '''
        Convolutional BLOCK-3
        '''
        self.C7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(3,3), padding=0, bias=False),
            nn.GroupNorm(1, 32), #Dividing the 32 channels into 1 group - Layer Normalization
            nn.ReLU()
            ) #output size = 6
        self.C8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=0, bias=False),
            nn.GroupNorm(1,32), #Dividing the 32 channels into 1 group - Layer Normalization
            nn.ReLU()
            ) #output size = 4
        self.c9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1), padding=0, bias=False),
            nn.GroupNorm(1,32), #Dividing the 32 channels into 1 group - Layer Normalization
            nn.ReLU()
            ) #output size = 4
        
        '''
        Global Average Pooling
        '''
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output size = 1
        
        self.C10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            )
    
    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.c3(x)
        x = self.pool1(x)
        
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        x = self.c6(x)
        x = self.pool2(x)
        
        x = self.C7(x)
        x = self.C8(x)
        x = self.c9(x)
        x = self.gap(x)
        
        x = self.C10(x)
        
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

'''
Model-3

Applied Batch Normalization
'''

class Model_3(nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()
        '''
        Convolutional BLOCK-1
        '''
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(32), 
            nn.ReLU()
            ) #output size = 32
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d( 32),
            nn.ReLU()
            ) #output size = 32
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
            ) #output size = 32
        '''
        Transition BLOCK-1
        '''
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 16
        
        '''
        Convolutional BLOCK-2
        '''
        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
            ) #output size = 16
        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
            ) #output size = 16
        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
            ) #output size = 16
        self.c6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
            ) #output size = 16
        '''
        Transition BLOCK-2
        '''
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 8
        
        '''
        Convolutional BLOCK-3
        '''
        self.C7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(3,3), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
            ) #output size = 6
        self.C8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
            ) #output size = 4
        self.c9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
            ) #output size = 4
        
        '''
        Global Average Pooling
        '''
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output size = 1
        
        self.C10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            )
    
    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.c3(x)
        x = self.pool1(x)
        
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        x = self.c6(x)
        x = self.pool2(x)
        
        x = self.C7(x)
        x = self.C8(x)
        x = self.c9(x)
        x = self.gap(x)
        
        x = self.C10(x)
        
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)