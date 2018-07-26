import sys
sys.path.insert(0, '../')

import torch
from torch import nn
from view import *
from join_table import *
from holder import *


class Adagrad:
	def __init__(self, m, opt):
		self.layer_etas = []
		self.vars = []
		# self.stds = []
		self.lr = opt.learning_rate
		self.m = m

		for p in m.parameters():
			self.vars.append(torch.Tensor().type_as(p.data).resize_as_(p.data).zero_().add(0.1))
			# self.stds.append(torch.Tensor().type_as(p.data).resize_as_(p.data))

	def step(self, shared):
		for i, p in enumerate(self.m.parameters()):
			# get the averaged gradient
			grad = p.grad.data.div(shared.batch_l)
			# adagrad step
			self.vars[i].addcmul_(1.0, grad, grad)
			# rstd = self.vars[i].rsqrt()
			# p.data.addcmul_(-self.lr, grad, rstd)
			std = self.vars[i].sqrt()
			p.data.addcdiv_(-self.lr, grad, std)

	def cuda(self):
		for i in xrange(len(self.vars)):
			self.vars[i] = self.vars[i].cuda()
			
			



