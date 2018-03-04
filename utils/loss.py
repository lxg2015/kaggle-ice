import torch
import torch.nn as nn
import torch.nn.functional as F

# class CrossEntropy(nn.Module):
#     def __init__(self):
#         super(CrossEntropy, self).__init__()
    
#     def forward(self, out, label):
#         out = out.squeeze()
#         label = label.float()
#         # print(out.size(), out, label.size(), label)
        
#         return F.binary_cross_entropy(out, label)
class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, out, label):
        out = out.squeeze()
        label = label.float()
        # print(out.size(), out, label.size(), label)
        return F.binary_cross_entropy_with_logits(out, label)#, size_average=False)

        # out = out.sigmoid()
        # return F.mse_loss(out, label)

class CrossEntropyWeight(nn.Module):
    def __init__(self):
        super(CrossEntropyWeight, self).__init__()
        weight = torch.Tensor([753./851, 1.])
        weight = weight.cuda()
        self.loss = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self, out, label):
        out = out.squeeze()
        return self.loss(out, label)


class fcnLoss(nn.Module):
    def __init__(self):
        super(fcnLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(size_average=True, ignore_index=255)
    
    def forward(self, out, target):
        print(out.shape, target.shape)
        n,w,h = out.shape
        target = target.view(n, -1)
        loss = self.criterion(out, target)
        return loss

