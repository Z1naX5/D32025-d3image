import torch.nn as nn
import torch
from d3net import D3net


class Model(nn.Module):
    def __init__(self,cuda=True):
        super(Model, self).__init__()
        self.model = D3net()
        if cuda:
            self.model.cuda()
        init_model(self)

    def forward(self, x, rev=False):
        if not rev:
            out = self.model(x)
        else:
            out = self.model(x, rev=True)
        return out


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = 0.01 * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)
