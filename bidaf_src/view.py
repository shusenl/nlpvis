import torch
from torch import nn

class View(nn.Module):
	def __init__(self, *argv):
		super(View, self).__init__()
		self.dims = argv

	def forward(self, x):
		return x.view(self.dims)