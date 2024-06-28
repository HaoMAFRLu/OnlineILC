"""Classes for different types of neural networks
"""
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class CNN_SEQ(nn.model):
    def __init__(self, in_channel: int, height: int,
                 width: int, filter_size: int, outputs: int) -> None:
        """Create the CNN with sequential inputs and outputs
        """
        super().__init__()
        l = height
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channel,
                                             out_channels=2*in_channel,
                                             kernel_size=(filter_size, 1),
                                             stride=(1, 1),
                                             padding=(0, 0),
                                             bias=True),
                                    nn.Tanh()
                                  )
        self.bn1 = nn.BatchNorm2d(num_features=2*in_channel)
        l = int((l-filter_size)/1+1)  # 31 x 8
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=2*in_channel,
                                             out_channels=2*in_channel,
                                             kernel_size=(filter_size, 1),
                                             stride=(1, 1),
                                             padding=(0, 0),
                                             bias=True),
                                    nn.Tanh()
                                   )
        self.bn2 = nn.BatchNorm2d(num_features=2*in_channel)
        l = int((l-filter_size)/1+1)  # 21 * 6
        self.fc = nn.Sequential(nn.Linear(2*in_channel*l*1, 8, bias=True) ,  
                                nn.Tanh(),           
                                nn.Linear(8, outputs, bias=True),
                                # nn.Tanh()
                                )
    
    def get_size(self, size, filter, stride, padding):
        def cal_size(h, f, s, p):
            return (h-f+2*p)/s+1
        height = cal_size(size[0], filter[0], stride[0], padding[0])
        width = cal_size(size[1], filter[1], stride[1], padding[1])
        return (height, width)

    def forward(self, inputs):
        preds = None
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        batch_size, channels, height, width = out.shape 
        out = out.view(-1, channels*height*width)
        preds = self.fc(out)
        return preds.float()
        

