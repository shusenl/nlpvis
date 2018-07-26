import sys

import torch
from torch import nn
from torch.autograd import Variable
from view import *
from holder import *
from util import *

class HighwayLayer(torch.nn.Module):
	def __init__(self, opt, shared):
		super(HighwayLayer, self).__init__()
		self.opt = opt
		self.shared = shared

		self.linear = nn.Linear(opt.hidden_size, opt.hidden_size*2)
		self.transform_activation = nn.ReLU()
		self.gate_activation = nn.Sigmoid()
		self.one = Variable(torch.ones(1), requires_grad=False)
		if opt.gpuid != -1:
			self.one = self.one.cuda()

	# seq is of shape (seq_l, word_vec_size
	def forward(self, seq):
		proj = self.linear(seq)
		orig = seq

		transform = self.transform_activation(proj[:, 0:self.opt.hidden_size])
		gate = self.gate_activation(proj[:, self.opt.hidden_size:self.opt.hidden_size*2])

		return gate * orig + (self.one - gate) * transform


	def post_init(self):
		self.linear.bias[self.opt.hidden_size:].data.fill_(1.0)


# Highway networks
class Highway(torch.nn.Module):
	def __init__(self, opt, shared):
		super(Highway, self).__init__()
		self.opt = opt
		self.shared = shared

		hw_layer = opt.hw_layer
		self.hw_layers = nn.ModuleList([HighwayLayer(opt, shared) for _ in xrange(hw_layer)])

	# input is encoding tensor of shape (seq_l, word_vec_size)
	def forward(self, seq):
		self.update_context()

		for hl in self.hw_layers:
			seq = hl(seq)

		return seq


	def update_context(self):
		pass

	def post_init(self):
		for layer in self.hw_layers:
			layer.post_init()


	def get_param_dict(self, root):
		is_cuda = self.opt.gpuid != -1
		param_dict = {}
		for n, p in self.hw_layers.named_parameters():
			param_dict['{0}.hw_layers.{1}'.format(root, n)] = torch2np(p.data, is_cuda)
		
		return param_dict


	def set_param_dict(self, param_dict, root):
		for n, p in self.hw_layers.named_parameters():
			p.data.copy_(torch.from_numpy(param_dict['{0}.hw_layers.{1}'.format(root, n)][:]))


	def begin_pass(self):
		pass

	def end_pass(self):
		pass

