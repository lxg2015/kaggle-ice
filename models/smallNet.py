import torch
import torch.nn as nn
import torch.nn.functional as F
from .basicModule import BasicModule
import random

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

class InceptionA(nn.Module):
    '''
    input 16 output 32, size/2
    '''
    def __init__(self):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(16, 16, 1, 1)
        self.branch3x3 = nn.Conv2d(16, 16, 3, 1, padding=1)
        self.branch5x5 = nn.Conv2d(16, 16, 5, 1, padding=2)
        self.branchpool = nn.MaxPool2d(3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.reduce = nn.Conv2d(64, 32, 3, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        h1 = self.branch1x1(x)
        h2 = self.branch3x3(x)
        h3 = self.branch5x5(x)
        h4 = self.branchpool(x)

        h = torch.cat([h1, h2, h3, h4], dim=1)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.reduce(h)
        h = self.bn2(h)
        h = self.relu2(h)

        return h

class InceptionI(nn.Module):
    '''
    input 2 output16, size
    '''
    def __init__(self):
        super(InceptionI, self).__init__()
        self.branch3x3 = nn.Conv2d(2, 5, 3, 1, padding=1)
        self.branch5x5 = nn.Conv2d(2, 5, 5, 1, padding=2)
        self.branch7x7 = nn.Conv2d(2, 6, 7, 1, padding=3)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h1 = self.branch3x3(x)
        h2 = self.branch5x5(x)
        h3 = self.branch7x7(x)
        h = torch.cat([h1, h2, h3], dim=1)
        h = self.bn(h)
        h = self.relu(h)
        
        return h

class smallNet(BasicModule):
    '''
    in 20x20
    '''
    def __init__(self, path):
        self.inplanes = 16
        super(smallNet, self).__init__(path)
        block = BasicBlock

        # self.convi = InceptionI()

        self.conv1 = nn.Conv2d(2, 16, 7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.LeakyReLU() #nn.ReLU()

        self.layer1 = self._make_layer(block, 16, 1, stride=1) # 16->16
        self.layer2 = self._make_layer(block, 32, 2, stride=1) # receptive field in output feature map 7
        # self.layer2 = InceptionA(); self.inplanes=32
        self.layer3 = self._make_layer(block, 64, 2, stride=1) # 11
        # self.layer1 = self._make_sequential(16, 16, 1, 1)
        # self.layer2 = self._make_sequential(16, 32, 2, 1)
        # self.layer3 = self._make_sequential(32, 64, 2, 1)
        self.layer4 = self._make_layer(block, 128, 2, stride=2) # 19
        # self.avgpool = nn.MaxPool2d(2)    # input 30 8 , 20 5
        self.fc1 = nn.Linear(128*5*5,128)
        self.fc1bn = nn.BatchNorm2d(128)
        self.fc1relu = nn.ReLU()
        self.fc2 = nn.Linear(128+1, 1)
        # self.maxpool1 = nn.MaxPool2d(3, 2, padding=1)
        self.conv5 = nn.Conv2d(128,128, 3, 1, padding=1)

        self.fp1 = nn.FractionalMaxPool2d(3, output_ratio=(0.8, 0.8))
        self.fp2 = nn.FractionalMaxPool2d(3, output_ratio=(0.8, 0.8))
        self.fp3 = nn.FractionalMaxPool2d(3, output_ratio=(0.8, 0.8))
        # self.fp4 = nn.FractionalMaxPool2d(3, output_ratio=(0.8, 0.8))
        
        
        
        self.apply(weight_init)
    
    def _make_sequential(self, inc, outc, stride, pad):
        return nn.Sequential(
            nn.Conv2d(inc, outc, 3, stride=stride, padding=pad),
            nn.BatchNorm2d(outc),
            nn.Sigmoid()
        )

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
        # h = self.stn(h)
        # h = self.bn0(h)
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu1(h)
        # h = self.maxpool1(h)
        # h = self.convi(h)

        h = self.layer1(h)
        h = self.fp1(h)
        h = self.layer2(h)
        h = self.fp2(h)
        
        # print(h.shape)
        # h_laternal = self.maxpool1(h)
        h = self.layer3(h) # 5,5
        h = self.fp3(h)
        # h_out = h
        # _,_,H,W = h.shape
        # print('lateral', h_laternal.shape) 
        h = self.layer4(h)
        # print(h.shape)
        # h = self.fp4(h)
        # h_laternal = F.upsample(h, size=(H,W), mode='bilinear')

        # h = torch.cat([h_out, h_laternal], dim=1)
        # h = self.conv5(h)
        # print('conv', h.size())
        
        # h = self.avgpool(h)
        h = h.view(h.size(0), -1)
        h = self.fc1(h)
        h = self.fc1bn(h)
        h = self.fc1relu(h)
        h = torch.cat([h, incs], dim=1)
        h = self.fc2(h)
        # print(h)
        if self.train:
            h += random.random()/5

        return h


def weight_init(m):
    # print(type(m), m)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal(m.weight)
        nn.init.kaiming_uniform(m.weight)    

if __name__ == '__main__':
    from torch.autograd import Variable
    import os
    path = os.path.expanduser('~/codedata/ice/')

    model = smallNet(path)
    data = Variable(torch.randn(1,2,40,40)) 
    incs = Variable(torch.randn(1,1)) 
    
    out = model(data, incs)
    print('loc', out.size(), out)
	