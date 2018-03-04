import torch
import torchvision
import torch.nn as nn

def Res18():
    resnet18 = torchvision.models.resnet18(pretrained=True)
    # resnet18 = torchvision.models.resnet18(pretrained=False)
    resnet18.conv1 = nn.Conv2d(2, 64, kernel_size=7, 
                        stride=2, padding=3, bias=False)
    resnet18.fc = nn.Linear(512, 2)
    # resnet18.fc = nn.Sequential(
    #     nn.Dropout(),
    #     nn.Linear(512, 1),
    #     nn.Sigmoid()
    # )
    
    # resnet18.fc = nn.Sequential(
    #     nn.Dropout(),
    #     nn.Linear(512, 64),
    #     nn.Dropout(),
    #     nn.ReLU(),
    #     nn.Linear(64, 1),
    #     nn.Sigmoid()
    # )

    # resnet18.fc = nn.Sequential(
    #     nn.Dropout(),
    #     nn.Linear(512, 64),
    #     nn.Dropout(),
    #     nn.ReLU(),
    #     nn.Linear(64, 2),
    # )
    return resnet18

if __name__ == '__main__':
    from torch.autograd import Variable

    model = Res18()
    data = Variable(torch.randn(1,3,224,224)) 
    out = model(data)
    print('loc', out.size(), out)
	