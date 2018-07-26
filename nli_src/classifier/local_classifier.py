import sys
# sys.path.insert(0, '../')

import torch
from torch import nn
from ..join_table import *
from ..view import *
from ..holder import *

class LocalClassifier(torch.nn.Module):
	def __init__(self, opt, shared):
		super(LocalClassifier, self).__init__()

		# temp stuff will be changed on the fly
		batch_l = 1
		sent_l1 = 2
		sent_l2 = 3

		cat_size = opt.hidden_size * 2
		self.input_view1 = View(batch_l * sent_l1, cat_size)
		self.input_view2 = View(batch_l * sent_l2, cat_size)
		self.input_unview1 = View(batch_l, sent_l1, opt.hidden_size)
		self.input_unview2 = View(batch_l, sent_l2, opt.hidden_size)
		self.input_joiner = JoinTable(2)
		self.phi_joiner = JoinTable(1)

		# bookkeeping
		self.shared = shared
		self.dropout = opt.dropout
		self.hidden_size = opt.hidden_size

		self.g = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(cat_size, opt.hidden_size),
			nn.ReLU(),
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.hidden_size),
			nn.ReLU())
		self.h = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(cat_size, opt.hidden_size),
			nn.ReLU(),
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.hidden_size),
			nn.ReLU(),
			nn.Linear(opt.hidden_size, opt.num_labels),
			nn.LogSoftmax(1))

	def init_weight_from(self, c):
		for i in [1,4]:
			self.g[i].weight.data.copy_(c.g[i].weight.data)
			if self.g[i].bias is not None and c.g[i].bias is not None:
				self.g[i].bias.data.copy_(c.g[i].bias.data)

		for i in [1,4,6]:
			self.h[i].weight.data.copy_(c.h[i].weight.data)
			if self.h[i].bias is not None and c.h[i].bias is not None:
				self.h[i].bias.data.copy_(c.h[i].bias.data)


	def forward(self, sent1, sent2, att1, att2):
		self.update_context()

		attended2 = att1.bmm(sent2)
		attended1 = att2.bmm(sent1)

		cat1 = self.input_joiner([sent1, attended2])
		cat2 = self.input_joiner([sent2, attended1])

		phi1 = self.input_unview1(self.g(self.input_view1(cat1)))
		phi2 = self.input_unview2(self.g(self.input_view2(cat2)))

		flat_phi1 = phi1.sum(1)
		flat_phi2 = phi2.sum(1)

		phi = self.phi_joiner([flat_phi1, flat_phi2])
		self.shared.out = self.h(phi)

		return self.shared.out

	def update_context(self):
		batch_l = self.shared.batch_l
		sent_l1 = self.shared.sent_l1
		sent_l2 = self.shared.sent_l2
		cat_size = self.hidden_size * 2

		self.input_view1.dims = (batch_l * sent_l1, cat_size)
		self.input_view2.dims = (batch_l * sent_l2, cat_size)
		self.input_unview1.dims = (batch_l, sent_l1, self.hidden_size)
		self.input_unview2.dims = (batch_l, sent_l2, self.hidden_size)

	def weights(self, m):
		classname = m.__class__.__name__
		if hasattr(m, 'weight'):
			print('{0} weight {1}'.format(classname, m.weight))
		if hasattr(m, 'bias'):
			print('{0} bias {1}'.format(classname, m.bias))

	def get_param_dict(self, root):
		is_cuda = self.opt.gpuid != -1
		param_dict = {}
		for i in [1,4]:
			param_dict['{0}.g[{1}].weight'.format(root, i)] = torch2np(self.g[i].weight.data, is_cuda)
			if self.g[i].bias is not None:
				param_dict['{0}.g[{1}].bias'.format(root, i)] = torch2np(self.g[i].bias.data, is_cuda)

		for i in [1,4,6]:
			param_dict['{0}.h[{1}].weight'.format(root, i)] = torch2np(self.h[i].weight.data, is_cuda)
			if self.h[i].bias is not None:
				param_dict['{0}.h[{1}].bias'.format(root, i)] = torch2np(self.h[i].bias.data, is_cuda)
		return param_dict


	def set_param_dict(self, param_dict, root):
		for i in [1,4]:
			self.g[i].weight.data.copy_(torch.from_numpy(param_dict['{0}.g[{1}].weight'.format(root, i)][:]))
			if self.g[i].bias is not None:
				self.g[i].bias.data.copy_(torch.from_numpy(param_dict['{0}.g[{1}].bias'.format(root, i)][:]))

		for i in [1,4,6]:
			self.h[i].weight.data.copy_(torch.from_numpy(param_dict['{0}.h[{1}].weight'.format(root, i)][:]))
			if self.h[i].bias is not None:
				self.h[i].bias.data.copy_(torch.from_numpy(param_dict['{0}.h[{1}].bias'.format(root, i)][:]))


if __name__ == '__main__':
	sys.path.insert(0, '../attention/')
	from torch.autograd import Variable
	from local_attention import *

	opt = Holder()
	opt.hidden_size = 3
	opt.dropout = 0.0
	opt.num_labels = 3
	shared = Holder()
	shared.batch_l = 2
	shared.sent_l1 = 5
	shared.sent_l2 = 8
	shared.input1 = Variable(torch.randn(shared.batch_l, shared.sent_l1, opt.hidden_size), True)
	shared.input2 = Variable(torch.randn(shared.batch_l, shared.sent_l2, opt.hidden_size), True)

	# build network
	attender = LocalAttention(opt, shared)
	classifier = LocalClassifier(opt, shared)

	# update batch info
	shared.batch_l = 2
	shared.sent_l1 = 5
	shared.sent_l2 = 8

	# run network
	shared.att1, shared.att2 = attender(shared.input1, shared.input2)
	shared.out = classifier(shared.input1, shared.input2, shared.att1, shared.att2)

	print(shared.att1)
	print(shared.att1.sum(2))
	print(shared.att2)
	print(shared.att2.sum(2))
	print(shared.out)
	#print(classifier)
	#print(classifier.g[1].weight)
	#print(classifier.g[1].bias)
	#classifier.apply(classifier.weights)
#
	#for i, p in enumerate(classifier.parameters()):
	#	print(p.data)
	#	print(p.grad)
	#
