import sys
# sys.path.insert(0, './encoder')
# sys.path.insert(0, './attention')
# sys.path.insert(0, './classifier')

import torch
from torch import nn
from torch import cuda
from view import *
from join_table import *
from holder import *
# from proj_encoder import *
from encoder import *
# from local_attention import *
from attention import *
# from local_classifier import *
from classifier import *
from torch.autograd import Variable
import numpy as np
from optimizer import *
import time

class Pipeline(torch.nn.Module):
	def __init__(self, opt, shared):
		super(Pipeline, self).__init__()

		self.shared = shared
		self.opt = opt

		if opt.encoder == 'proj':
			self.encoder = ProjEncoder(opt, shared)
		else:
			raise Exception('unrecognized enocder: {0}'.format(opt.encoder))

		if opt.attention == 'local':
			self.attention = LocalAttention(opt, shared)
		else:
			raise Exception('unrecognized attention: {0}'.format(opt.attention))

		if opt.classifier == 'local':
			self.classifier = LocalClassifier(opt, shared)
		else:
			raise Exception('unrecognized classifier: {0}'.format(opt.classifier))

	def init_weight(self):
		if self.opt.param_init_type == 'xavier_uniform':
			for n, p in self.named_parameters():
				if 'weight' in n:
					nn.init.xavier_uniform(p)
				elif 'bias' in n:
					nn.init.constant(p, 0)
		elif self.opt.param_init_type == 'uniform':
			for n, p in self.named_parameters():
				if 'weight' in n:
					p.data.copy_(torch.randn(p.data.shape)).mul_(self.opt.param_init)
				elif 'bias' in n:
					p.data.copy_(torch.randn(p.data.shape)).mul_(self.opt.param_init)

	# init weight form a pretrained model
	#	will recursively pass down network subgraphs accordingly
	def init_weight_from(self, m):
		self.encoder.init_weight_from(m.encoder)
		self.attention.init_weight_from(m.attention)
		self.classifier.init_weight_from(m.classifier)


	def forward(self, sent1, sent2):
		shared = self.shared
		shared.input_enc1, shared.input_enc2 = self.encoder(sent1, sent2)
		shared.att1, shared.att2 = self.attention(shared.input_enc1, shared.input_enc2)
		shared.out, shared.flat_phi1, shared.flat_phi2 = self.classifier(shared.input_enc1, shared.input_enc2, shared.att1, shared.att2)

		# if there is any fwd pass hooks, execute them
		if hasattr(self.opt, 'forward_hooks') and self.opt.forward_hooks != '':
			run_forward_hooks(self.opt, self.shared, self)

		return shared.out


	# call this explicitly
	def update_context(self, batch_ex_idx, batch_l, sent_l1, sent_l2, res_map=None):
		self.shared.batch_ex_idx = batch_ex_idx
		self.shared.batch_l = batch_l
		self.shared.sent_l1 = sent_l1
		self.shared.sent_l2 = sent_l2
		self.shared.res_map = res_map

	def get_param_dict(self):
		param_dict = {}
		param_dict.update(self.encoder.get_param_dict('encoder'))
		param_dict.update(self.attention.get_param_dict('attention'))
		param_dict.update(self.classifier.get_param_dict('classifier'))
		return param_dict

	def set_param_dict(self, param_dict):
		self.encoder.set_param_dict(param_dict, 'encoder')
		self.attention.set_param_dict(param_dict, 'attention')
		self.classifier.set_param_dict(param_dict, 'classifier')


def overfit():
	sys.path.insert(0, '../attention/')

	opt = Holder()
	opt.gpuid = 1
	opt.word_vec_size = 3
	opt.hidden_size = 4
	opt.dropout = 0.0
	opt.num_att_labels = 1
	opt.num_labels = 3
	opt.encoder = 'proj'
	opt.attention = 'labeled_local_hard'
	opt.classifier = 'labeled_local'
	opt.learning_rate = 0.05
	opt.param_init = 0.01
	shared = Holder()
	shared.batch_l = 2
	shared.sent_l1 = 5
	shared.sent_l2 = 8

	input1_ = torch.randn(shared.batch_l, shared.sent_l1, opt.word_vec_size)
	input2_ = torch.randn(shared.batch_l, shared.sent_l2, opt.word_vec_size)
	gold_ = torch.from_numpy(np.random.randint(opt.num_labels, size=shared.batch_l))
	if opt.gpuid != -1:
		input1_ = input1_.cuda()
		input2_ = input2_.cuda()
		gold_ = gold_.cuda()
	shared.input1 = Variable(input1_, True)
	shared.input2 = Variable(input2_, True)
	gold = Variable(gold_, False)

	# build network
	m = Pipeline(opt, shared)
	m.init_weight()
	criterion = torch.nn.NLLLoss(size_average=False)
	optim = Adagrad(m, opt)

	if opt.gpuid != -1:
		m = m.cuda()
		criterion = criterion.cuda()
		optim.cuda()

	# update batch info
	shared.batch_l = 2
	shared.sent_l1 = 5
	shared.sent_l2 = 8

	# run network
	shared.out = m(shared.input1, shared.input2)
	loss = criterion(shared.out, gold)
	print(shared.out)
	print(loss)
	print(m)
	for i, p in enumerate(m.parameters()):
		print(p.data)

	print(m.state_dict())


	for i in xrange(300):
		print('epoch: {0}'.format(i+1))

		shared.out = m(shared.input1, shared.input2)
		loss = criterion(shared.out, gold)
		print('y\': {0}'.format(shared.out.exp()))
		print('y*: {0}'.format(gold))
		print('loss: {0}'.format(loss))

		m.zero_grad()
		loss.backward()
		optim.step(shared)

if __name__ == '__main__':
	overfit()
