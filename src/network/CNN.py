"""Classes for different types of neural networks
"""
import torch.nn as nn
import torch 
import torchvision.models as models


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
                                             out_channels=16*in_channel,
                                             kernel_size=(filter_size, 1),
                                             stride=(1, 1),
                                             padding=(0, 0),
                                             bias=True),
                                    nn.ReLU()
                                   )
        self.bn3 = nn.BatchNorm2d(num_features=16*in_channel)
        l = int((l - filter_size)/1 + 1)

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=16*in_channel,
                                             out_channels=64*in_channel,
                                             kernel_size=(filter_size, 1),
                                             stride=(1, 1),
                                             padding=(0, 0),
                                             bias=True),
                                    nn.ReLU()
                                   )
        self.bn4 = nn.BatchNorm2d(num_features=64*in_channel)
        l = int((l - filter_size)/1 + 1)

        self.fc = nn.Sequential(nn.Linear(64*in_channel*l, 128, bias=True) ,  
                                nn.ReLU(),           
                                nn.Linear(128, output_dim, bias=True)
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

        out = self.conv4(out)
        out = self.bn4(out)

        batch_size, channels, height, width = out.shape 
        out = out.view(-1, channels*height*width)
        preds = self.fc(out)
        return preds.float()
        
class SimplifiedResNet(nn.Module):
    """Simplified ResNet
    """
    def __init__(self, num_classes=550):
        super(SimplifiedResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(16, 16, 2)
        self.layer2 = self._make_layer(16, 32, 2, stride=2)
        self.layer3 = self._make_layer(32, 64, 2, stride=2)
        self.layer4 = self._make_layer(64, 128*2, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128*2, num_classes)
    
    def _make_layer(self, in_planes, out_planes, blocks, stride=1):
        layers = []
        layers.append(self._basic_block(in_planes, out_planes, stride))
        for _ in range(1, blocks):
            layers.append(self._basic_block(out_planes, out_planes))
        return nn.Sequential(*layers)
    
    def _basic_block(self, in_planes, out_planes, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.float()

class CustomResNet18(nn.Module):
    def __init__(self, input_channels=1, num_classes=550, pretrained=True):
        super(CustomResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        
        if input_channels != 3:
            self.model.conv1 = nn.Conv2d(input_channels, self.model.conv1.out_channels, 
                                         kernel_size=7, stride=2, padding=3, bias=False)
            
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x).float()
