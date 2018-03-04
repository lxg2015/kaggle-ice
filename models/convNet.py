import torch
import torch.nn as nn
import torch.nn.functional as F
from .basicModule import BasicModule
import random
# from modules import ConvOffset2d

# dropout example

class convNet(BasicModule):
    def __init__(self, path):
        super(convNet, self).__init__(path)

        # spatial transformer network
        self.localization = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            # nn.Conv2d(16, 32, kernel_size=3, stride=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.MaxPool2d(2, stride=2),
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(16*8*8, 16),
            nn.ReLU(True),
            nn.Linear(16, 3*2),
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # self.compute_offset1 = nn.Conv2d(128, 2*3*3, 3, 1, padding=1, bias=False)
        # self.apply_offset1 = ConvOffset2d(128, 128, 3, 1, padding=1, 
        #                             num_deformable_groups=1)
        # self.compute_offset2 = nn.Conv2d(256, 2*3*3, 3, 1, padding=1, bias=False)
        # self.apply_offset2 = ConvOffset2d(256, 256, 3, 1, padding=1, 
        #                             num_deformable_groups=1)
        # self.compute_offset0 = nn.Conv2d(32, 2*3*3, 3, 1, padding=1, bias=False)
        # self.apply_offset0 = ConvOffset2d(32, 32, 3, 1, padding=1, 
        #                             num_deformable_groups=1)                                    
        self.gpool = nn.MaxPool2d(5)
        
        self.fc1 = nn.Linear(256,128, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU(inplace=True)
        # self.drop1 = nn.Dropout2d(p=0.5)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm2d(64) 
        self.relu2 = nn.ReLU(inplace=True)
        # self.drop2 = nn.Dropout2d()

        self.fc3 = nn.Linear(64,1)
        self.sigmoid = nn.Sigmoid()

        # init
        self.apply(weight_init)

        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1,0,0,0,1,0])
    
    def forward(self, x, incs):
        out = x
        out = self.stn(out)

        out = self.layer1(out) # 2->32
        # offset = self.compute_offset0(out)
        # out = self.apply_offset0(out, offset)
        
        out = self.layer2(out)
        out = self.layer3(out) # 128->128
        
        # offset = self.compute_offset1(out)
        # out = self.apply_offset1(out, offset)
        
        out = self.layer4(out)
        out = self.layer5(out)

        # deformable conv
        # offset = self.compute_offset2(out)
        # out = self.apply_offset2(out, offset)

        out = self.gpool(out)
        # print('conv:', out.size())

        out = out.view(out.size(0), -1)

        # out = torch.cat((out, incs), dim=1) # add inc_angle
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        # out = self.drop2(out)
        
        out = self.fc3(out)
        # print(out)
        if self.train:
            out = out + random.random()/5  # add noise(0,0.2)
        
        # with binary_cross_entropy_with_logits, sigmoid is not neccessay
        # out = self.sigmoid(out) 

        return out
    
    def stn(self, x):
        xs = self.localization(x)
        # print(xs.size())
        xs = xs.view(x.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        # print('theta', theta.size())

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

def weight_init(m):
    # print(type(m), m)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal(m.weight)
        nn.init.kaiming_uniform(m.weight)    

if __name__ == '__main__':
    from torch.autograd import Variable

    model = convNet('sd')
    data = Variable(torch.randn(3,2,75,75))
    result = model(data)
    print('data: ', data.size())
    print('result: ', result.size())