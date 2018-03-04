import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .basicModule import BasicModule

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class lateralNet(BasicModule):
    '''
    in 40x40
    '''
    def __init__(self, path):
        self.inplanes = 16
        super(lateralNet, self).__init__(path)
        block = BasicBlock
        
        self.conv1 = nn.Conv2d(2, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(True)

        self.layer1 = self._make_layer(block, 16, 1, stride=1)
        self.layer2 = self._make_layer(block, 32, 2, stride=2) # receptive field in output feature map 7
        self.layer3 = self._make_layer(block, 64, 2, stride=2) # 11
        self.layer4 = self._make_layer(block, 128, 2, stride=2) # 19
        self.layer5 = self._make_layer(block, 128, 2, stride=2) # 35
        
        self.conv6 = nn.Conv2d(256, 192, 3, 1, padding=1)
        self.avgpool = nn.MaxPool2d(3)   
             
        self.fc2 = nn.Linear(192, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1) #(9,9)

    def _make_layer(self, block, outplanes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != outplanes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, outplanes*block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes*block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, outplanes, stride, downsample))
        
        self.inplanes = outplanes * block.expansion # first resblock out channels
        for i in range(1, blocks):
            layers.append(block(self.inplanes, outplanes))
        return nn.Sequential(*layers)

    def forward(self, x, incs):
        h = x
        # h = self.bn0(h)
        h = self.conv1(h)
        h = self.relu1(self.bn1(h))
        # h = self.maxpool1(h)

        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h_laternal = self.maxpool1(h) # 64
        # print('lateral', h_laternal.shape) 
        h = self.layer5(h) # (5,5)
        # print('h', h.shape)
        _,_,H,W = h_laternal.shape
        h_up = F.upsample(h, size=(H,W), mode='bilinear')
        h = torch.cat([h_up, h_laternal], dim=1) # 192x9x9
        h = self.conv6(h)
        # print('conv', h.size())
        h = self.avgpool(h)
        h = h.view(h.size(0), -1)

        h = self.fc2(h)
        # print(h)
        if self.train:
            h += random.random()/5

        return h

if __name__ == '__main__':
    from torch.autograd import Variable
    import os
    path = os.path.expanduser('~/codedata/ice/')

    model = lateralNet(path)
    data = Variable(torch.randn(1,2,75,75)) 
    incs = Variable(torch.randn(1,1)) 
    
    out = model(data, incs)
    print('loc', out.size())
