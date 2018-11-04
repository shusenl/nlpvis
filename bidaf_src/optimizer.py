import sys
import math
import torch
from torch import nn
from .view import *
from .join_table import *
from .holder import *
from .util import *


class Adagrad:
	def __init__(self, opt, shared):
		self.opt = opt
		self.shared = shared
		self.optim = None
		self.clip = opt.clip if opt.clip > 0.0  else 10000000000.0

	def step(self, m):
		params = [p for p in m.parameters() if p.requires_grad]
		if self.optim is None:
			self.optim = torch.optim.Adagrad(params, lr=self.opt.learning_rate)

		grad_norm2 = nn.utils.clip_grad_norm(params, self.clip, norm_type=2)

		self.optim.step()

		return grad_norm2

			
class Adam:
	def __init__(self, opt, shared):
		self.opt = opt
		self.shared = shared
		self.optim = None
		self.clip = opt.clip if opt.clip > 0.0  else 10000000000.0

	def step(self, m):
		params = [p for p in m.parameters() if p.requires_grad]
		if self.optim is None:
			self.optim = torch.optim.Adam(params, lr=self.opt.learning_rate, betas=(0.9, 0.9))

		grad_norm2 = nn.utils.clip_grad_norm(params, self.clip, norm_type=2)

		self.optim.step()

		return grad_norm2


class Adadelta:
	def __init__(self, opt, shared):
		self.opt = opt
		self.shared = shared
		self.optim = None
		self.clip = opt.clip if opt.clip > 0.0  else 10000000000.0

	def step(self, m):
		params = [p for p in m.parameters() if p.requires_grad]
		if self.optim is None:
			self.optim = torch.optim.Adadelta(params, lr=self.opt.learning_rate, rho=0.95)

		grad_norm2 = nn.utils.clip_grad_norm(params, self.clip, norm_type=2)

		self.optim.step()

		return grad_norm2




class Optimizer:
	def __init__(self, opt, shared):
		if opt.optim == 'adagrad':
			self.optim = Adagrad(opt, shared)
		elif opt.optim == 'adam':
			self.optim = Adam(opt, shared)
		elif opt.optim == 'adadelta':
			self.optim = Adadelta(opt, shared)
		else:
			print('unrecognized optim: {0}'.format(opt.optim))
			assert(False)
		self.__FLAG = False

	def step(self, m):
		if not self.__FLAG:
			noupdate_names = []
			for n,p in m.named_parameters():
				if not p.requires_grad or p.grad is None:
					noupdate_names.append(n)
			if len(noupdate_names) != 0:
				print('fields that do not have gradient: {0}'.format(noupdate_names))
			self.__FLAG = True

		return self.optim.step(m)


