"""Classes for different types of neural networks
"""
import torch.nn as nn

class CNN_SEQ(nn.Module):
    def __init__(self, in_channel: int, height: int,
                 width: int, filter_size: int, output_dim: int,
                 padding1: int=3, padding2: int=3) -> None:
        """Create the CNN with sequential inputs and outputs
        """
        super().__init__()
        l = height
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channel,
                                             out_channels=2*in_channel,
                                             kernel_size=(filter_size, 1),
                                             stride=(1, 1),
                                             padding=(padding1, 0),
                                             bias=True),
                                    nn.ReLU()
                                  )
        self.bn1 = nn.BatchNorm2d(num_features=2*in_channel)
        l = int((l+padding1+padding2 - filter_size)/1 + 1)

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=2*in_channel,
                                             out_channels=4*in_channel,
                                             kernel_size=(filter_size, 1),
                                             stride=(1, 1),
                                             padding=(0, 0),
                                             bias=True),
                                    nn.ReLU()
                                   )
        self.bn2 = nn.BatchNorm2d(num_features=4*in_channel)
        l = int((l - filter_size)/1 + 1)

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=4*in_channel,
                                             out_channels=8*in_channel,
                                             kernel_size=(filter_size, 1),
                                             stride=(1, 1),
                                             padding=(0, 0),
                                             bias=True),
                                    nn.ReLU()
                                   )
        self.bn3 = nn.BatchNorm2d(num_features=8*in_channel)
        l = int((l - filter_size)/1 + 1)

        self.fc = nn.Sequential(nn.Linear(8*in_channel*l, 32, bias=True) ,  
                                nn.ReLU(),           
                                nn.Linear(32, output_dim, bias=True)
                                # nn.Tanh()
                                )
    
    def forward(self, inputs):
        preds = None
        out = self.conv1(inputs)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        batch_size, channels, height, width = out.shape 
        out = out.view(-1, channels*height*width)
        preds = self.fc(out)
        return preds.float()
        

