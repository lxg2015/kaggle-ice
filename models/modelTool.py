import torch
import torch.nn as nn

def get_parameters(model, prelu=False):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            yield m.weight
            if m.bias:
                yield m.bias
        elif isinstance(m, nn.Linear):
            yield m.weight
            yield m.bias
        elif isinstance(m, nn.PReLU):
            yield m.weight
        else:
            print("module: %s 0 leraning rate" % str(m))
