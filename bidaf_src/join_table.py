import torch
from torch import nn

class JoinTable(nn.Module):
    def __init__(self, dim):
        super(JoinTable, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        y = torch.cat(x, self.dim)
        return y