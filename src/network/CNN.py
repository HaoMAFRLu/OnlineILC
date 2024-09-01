"""Classes for different types of neural networks
"""
import torch.nn as nn
import torch 
import torchvision.models as models
import math

class CNN_SEQ(nn.Module):
    def __init__(self, in_channel: int, height: int,
                 width: int, filter_size: int, output_dim: int) -> None:
        """Create the CNN with sequential inputs and outputs
        """
        super().__init__()
        l = height
        padding = math.floor(filter_size/2)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channel,
                                             out_channels=8*in_channel,
                                             kernel_size=filter_size,
                                             stride=(1, 1),
                                             padding=(padding, padding),
                                             bias=True),
                                    nn.ReLU()
                                  )
        l = int((l+2*padding - filter_size)/1 + 1)

        self.avg_pool1 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        self.bn1 = nn.BatchNorm2d(num_features=8*in_channel)
        

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=8*in_channel,
                                             out_channels=32*in_channel,
                                             kernel_size=filter_size,
                                             stride=(1, 1),
                                             padding=(0, padding),
                                             bias=True),
                                    nn.ReLU()
                                   )
        self.avg_pool2 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        self.bn2 = nn.BatchNorm2d(num_features=32*in_channel)
        l = int((l - filter_size)/1 + 1)

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32*in_channel,
                                             out_channels=128*in_channel,
                                             kernel_size=filter_size,
                                             stride=(1, 1),
                                             padding=(0, padding),
                                             bias=True),
                                    nn.ReLU()
                                   )
        self.avg_pool3 = nn.AvgPool2d(kernel_size=(3, 1), stride=(3, 1))
        self.bn3 = nn.BatchNorm2d(num_features=128*in_channel)
        l = int((l - filter_size)/1 + 1)

        # self.conv4 = nn.Sequential(nn.Conv2d(in_channels=32*in_channel,
        #                                      out_channels=128*in_channel,
        #                                      kernel_size=filter_size,
        #                                      stride=(1, 1),
        #                                      padding=(0, padding),
        #                                      bias=True),
        #                             nn.ReLU()
        #                            )
        # self.bn4 = nn.BatchNorm2d(num_features=128*in_channel)
        # l = int((l - filter_size)/1 + 1)

        self.fc = nn.Sequential(nn.Linear(128*in_channel*17, 128, bias=True) ,  
                                nn.ReLU(),           
                                nn.Linear(128, 64, bias=True),
                                nn.ReLU(),
                                nn.Linear(64, output_dim, bias=True),
                                )
    
    def forward(self, inputs):
        preds = None
        out = self.conv1(inputs)
        out = self.avg_pool1(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.avg_pool2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.avg_pool3(out)
        out = self.bn3(out)

        # out = self.conv4(out)
        # out = self.bn4(out)

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

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=2048, dropout=dropout, activation='relu')
        self.fc_out = nn.Linear(model_dim, output_dim)
    
    def forward(self, src, tgt):
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32))
        tgt = self.embedding(tgt) * torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32))
        out = self.transformer(src, tgt)
        out = self.fc_out(out)
        return out

