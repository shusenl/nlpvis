import sys
# sys.path.insert(0, '../')

import torch
from torch import nn
from ..view import *
from ..holder import *


class ProjEncoder(torch.nn.Module):
	def __init__(self, opt, shared):
		super(ProjEncoder, self).__init__()
		self.proj = nn.Linear(opt.word_vec_size, opt.hidden_size, False)

		# temp stuff will be changed on the fly
		batch_l = 1
		sent_l1 = 2
		sent_l2 = 3

		self.input_proj_view1 = View(batch_l * sent_l1, opt.word_vec_size)
		self.input_proj_view2 = View(batch_l * sent_l2, opt.word_vec_size)
		self.input_proj_unview1 = View(batch_l, sent_l1, opt.hidden_size)
		self.input_proj_unview2 = View(batch_l, sent_l2, opt.hidden_size)

		# bookkeeping
		self.input_size = opt.word_vec_size
		self.hidden_size = opt.hidden_size
		self.dropout = opt.dropout
		self.shared = shared

	def init_weight_from(self, e):
		self.proj.weight.data.copy_(e.proj.weight.data)
		if self.proj.bias is not None and e.proj.bias is not None:
			self.proj.bias.data.copy_(e.proj.bias.data)

	def forward(self, sent1, sent2):
		self.update_context()

		self.shared.input_enc1 = self.input_proj_unview1(self.proj(self.input_proj_view1(sent1)))
		self.shared.input_enc2 = self.input_proj_unview2(self.proj(self.input_proj_view2(sent2)))
		return [self.shared.input_enc1, self.shared.input_enc2]

	def update_context(self):
		batch_l = self.shared.batch_l
		sent_l1 = self.shared.sent_l1
		sent_l2 = self.shared.sent_l2
		input_size = self.input_size
		hidden_size = self.hidden_size

		self.input_proj_view1.dims = (batch_l * sent_l1, input_size)
		self.input_proj_view2.dims = (batch_l * sent_l2, input_size)
		self.input_proj_unview1.dims = (batch_l, sent_l1, hidden_size)
		self.input_proj_unview2.dims = (batch_l, sent_l2, hidden_size)

	def get_param_dict(self, root):
		is_cuda = self.opt.gpuid != -1
		param_dict = {}
		param_dict['{0}.proj.weight'.format(root)] = torch2np(self.proj.weight.data, is_cuda)
		if self.proj.bias is not None:
			param_dict['{0}.bias'.format(root)] = torch2np(self.proj.bias.data, is_cuda)
		return param_dict

	def set_param_dict(self, param_dict, root):
		self.proj.weight.data.copy_(torch.from_numpy(param_dict['{0}.proj.weight'.format(root)][:]))
		if self.proj.bias is not None:
			self.proj.bias.data.copy_(torch.from_numpy(param_dict['{0}.proj.bias'.format(root)][:]))


if __name__ == '__main__':
	from torch.autograd import Variable

	opt = Holder()
	opt.input_size = 3
	opt.hidden_size = 4
	opt.dropout = 0.0
	shared = Holder()
	shared.batch_l = 1
	shared.sent_l1 = 5
	shared.sent_l2 = 8
	shared.input1 = Variable(torch.randn(shared.batch_l, shared.sent_l1, opt.input_size), True)
	shared.input2 = Variable(torch.randn(shared.batch_l, shared.sent_l2, opt.input_size), True)

	# build network
	encoder = ProjEncoder(opt, shared)

	# update batch info
	shared.batch_l = 1
	shared.sent_l1 = 5
	shared.sent_l2 = 8

	# run network
	rs = encoder(shared.input1, shared.input2)
	print(rs)
