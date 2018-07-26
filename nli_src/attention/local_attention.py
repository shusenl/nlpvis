import sys
# sys.path.insert(0, '../')

import torch
from torch import nn
from torch.autograd import Variable
from ..view import *
from ..holder import *

class LocalAttention(torch.nn.Module):
	def __init__(self, opt, shared):
		super(LocalAttention, self).__init__()
		self.f = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.hidden_size),
			nn.ReLU(),
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.hidden_size),
			nn.ReLU())

		# temp stuff will be changed on the fly
		batch_l = 1
		sent_l1 = 2
		sent_l2 = 3

		self.input_view1 = View(batch_l * sent_l1, opt.hidden_size)
		self.input_view2 = View(batch_l * sent_l2, opt.hidden_size)
		self.input_unview1 = View(batch_l, sent_l1, opt.hidden_size)
		self.input_unview2 = View(batch_l, sent_l2, opt.hidden_size)
		self.score_view1 = View(batch_l * sent_l1, sent_l2)
		self.score_view2 = View(batch_l * sent_l2, sent_l1)
		self.score_unview1 = View(batch_l, sent_l1, sent_l2)
		self.score_unview2 = View(batch_l, sent_l2, sent_l1)
		self.softmax = nn.Softmax(1)

		# bookkeeping
		self.shared = shared
		self.opt = opt
		self.dropout = opt.dropout
		self.hidden_size = opt.hidden_size

	def init_weight_from(self, a):
		for i in [1,4]:
			self.f[i].weight.data.copy_(a.f[i].weight.data)
			if self.f[i].bias is not None and a.f[i].bias is not None:
				self.f[i].bias.data.copy_(a.f[i].bias.data)


	def forward(self, sent1, sent2):
		self.update_context()

		hidden1 = self.input_unview1(self.f(self.input_view1(sent1)))
		hidden2 = self.input_unview2(self.f(self.input_view2(sent2)))
		# score tensors of size batch_l x sent_l1 x sent_l2
		score1 = hidden1.bmm(hidden2.transpose(1,2))
		score2 = score1.transpose(1,2).contiguous()
		self.shared.score1 = score1
		self.shared.score2 = score2
		# to modify score, use self.shared.score1.data[:] = whatever
		# 	doring so will modify the raw data in torch, thus dangerous.

		# to use externally specified att values and discard the computed ones
		if self.opt.customize_att == 1:
			print "###### using custoimized_att #######"
			# self.shared.customized_att* will be a torch tensor
			# thus wrap the tensor in Variable to proceed forward pass
			customized1 = self.shared.customized_att1 if self.opt.gpuid == -1 else self.shared.customized_att1.cuda()
			customized2 = self.shared.customized_att2 if self.opt.gpuid == -1 else self.shared.customized_att2.cuda()
			self.shared.att_soft1 = Variable(customized1, requires_grad=True)
			self.shared.att_soft2 = Variable(customized2, requires_grad=True)
			# print self.shared.att_soft1, self.shared.att_soft2
		else:
			self.shared.att_soft1 = self.score_unview1(self.softmax(self.score_view1(self.shared.score1)))
			self.shared.att_soft2 = self.score_unview2(self.softmax(self.score_view2(self.shared.score2)))
			# print self.shared.att_soft1, self.shared.att_soft2

		return [self.shared.att_soft1, self.shared.att_soft2]


	def update_context(self):
		batch_l = self.shared.batch_l
		sent_l1 = self.shared.sent_l1
		sent_l2 = self.shared.sent_l2
		hidden_size = self.hidden_size

		self.input_view1.dims = (batch_l * sent_l1, hidden_size)
		self.input_view2.dims = (batch_l * sent_l2, hidden_size)
		self.input_unview1.dims = (batch_l, sent_l1, hidden_size)
		self.input_unview2.dims = (batch_l, sent_l2, hidden_size)
		self.score_view1.dims = (batch_l * sent_l1, sent_l2)
		self.score_view2.dims = (batch_l * sent_l2, sent_l1)
		self.score_unview1.dims = (batch_l, sent_l1, sent_l2)
		self.score_unview2.dims = (batch_l, sent_l2, sent_l1)


	def get_param_dict(self, root):
		is_cuda = self.opt.gpuid != -1
		param_dict = {}
		for i in [1,4]:
			param_dict['{0}.f[{1}].weight'.format(root, i)] = torch2np(self.f[i].weight.data, is_cuda)
			if self.f[i].bias is not None:
				param_dict['{0}.f[{1}].bias'.format(root, i)] = torch2np(self.f[i].bias.data, is_cuda)
		return param_dict

	def set_param_dict(self, param_dict, root):
		for i in [1,4]:
			self.f[i].weight.data.copy_(torch.from_numpy(param_dict['{0}.f[{1}].weight'.format(root, i)][:]))
			if self.f[i].bias is not None:
				self.f[i].bias.data.copy_(torch.from_numpy(param_dict['{0}.f[{1}].bias'.format(root, i)][:]))


if __name__ == '__main__':
	from torch.autograd import Variable
	hidden_size = 3

	opt = Holder()
	opt.hidden_size = 3
	opt.dropout = 0.0
	shared = Holder()
	shared.batch_l = 1
	shared.sent_l1 = 5
	shared.sent_l2 = 8
	shared.input1  = Variable(torch.randn(shared.batch_l, shared.sent_l1, opt.hidden_size), True)
	shared.input2 = Variable(torch.randn(shared.batch_l, shared.sent_l2, opt.hidden_size), True)

	# build network
	attender = LocalAttention(opt, shared)

	# update batch info
	shared.batch_l = 1
	shared.sent_l1 = 5
	shared.sent_l2 = 8

	# run network
	rs = attender(shared.input1, shared.input2)
	print(rs)
	print(rs[0].sum(2))
	print(rs[1].sum(2))
