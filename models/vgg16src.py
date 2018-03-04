import torchvision
import torch
import torch.nn as nn

def VGG16():
    vgg16 = torchvision.models.vgg16(pretrained=True)
    vgg16.classifier = nn.Linear(25088, 2)

    
    return vgg16