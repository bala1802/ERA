import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Architecture:
    - C1, C2, C3, C4
    - Each block will have 3 Convolution layers.
    - First layer will be Depthwise Seperable convolution layer; 
    - Second Layer will be a 1X1
    - Third layer will be a convolution layer with dilation as 2
'''

class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        
        '''
        C1 BLOCK: c1, c2, c3
        '''
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=60, kernel_size=(3,3), padding=1, bias=False, groups=3),
            nn.BatchNorm2d(60),
            nn.ReLU()
            )
        self.c2 = nn.Sequential(
            nn.Conv2d(in_channels=60, out_channels=30, kernel_size=(1,1), padding=1, bias=False),
            nn.BatchNorm2d(30),
            nn.ReLU()
            )
        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=60, kernel_size=(3,3), dilation=2, padding=0, bias=False),
            nn.BatchNorm2d(60),
            nn.ReLU()
            )
        
        '''
        C2 BLOCK: c4, c5, c6
        '''
        self.c4 = nn.Sequential(
            nn.Conv2d(in_channels=60, out_channels=120, kernel_size=(3,3), padding=1, bias=False, groups=60),
            nn.BatchNorm2d(120),
            nn.ReLU()
            )
        self.c5 = nn.Sequential(
            nn.Conv2d(in_channels=120, out_channels=30, kernel_size=(1,1), padding=1, bias=False),
            nn.BatchNorm2d(30),
            nn.ReLU()
            )
        self.c6 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=60, kernel_size=(3,3), dilation=2, padding=0, bias=False),
            nn.BatchNorm2d(60),
            nn.ReLU()
            )
            
        '''
        C3 BLOCK: c7, c8, c9
        '''
        self.c7 = nn.Sequential(
            nn.Conv2d(in_channels=60, out_channels=120, kernel_size=(3,3), padding=1, bias=False, groups=60),
            nn.BatchNorm2d(120),
            nn.ReLU()
            )
        self.c8 = nn.Sequential(
            nn.Conv2d(in_channels=120, out_channels=30, kernel_size=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(30),
            nn.ReLU()
            )
        self.c9 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=60, kernel_size=(3,3), dilation=2, padding=0, bias=False),
            nn.BatchNorm2d(60),
            nn.ReLU()
            )
        
        '''
        C4 BLOCK: c10, c11, c12
        '''
        self.c10 = nn.Sequential(
            nn.Conv2d(in_channels=60, out_channels=120, kernel_size=(3,3), padding=0, bias=False, groups=60),
            nn.BatchNorm2d(120),
            nn.ReLU()
            )
        self.c11 = nn.Sequential(
            nn.Conv2d(in_channels=120, out_channels=30, kernel_size=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(30),
            nn.ReLU()
            )
        self.c12 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=60, kernel_size=(3,3), dilation=2, padding=0, bias=False),
            nn.BatchNorm2d(60),
            nn.ReLU()
            )
        
        '''
        Global Average Pooling
        '''
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=11)
        )
        
        self.c13 = nn.Sequential(
            nn.Conv2d(in_channels=60, out_channels=10, kernel_size=(1,1), padding=0, bias=False),
            )
            
    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        
        x = self.c7(x)
        x = self.c8(x)
        x = self.c9(x)
        
        x = self.c10(x)
        x = self.c11(x)
        x = self.c12(x)
        
        x = self.gap(x)
        x = self.c13(x)
        
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        