import sys

import torch
from torch import nn
from torch import cuda
import numpy as np
import time

from .optimizer import *
from .view import *
from .join_table import *
from .holder import *
from .embeddings import *
from .encoder import *
from .biattention import *
from .reencoder import *
from .classifier import *


class Pipeline(torch.nn.Module):
	def __init__(self, opt, shared):
		super(Pipeline, self).__init__()

		self.shared = shared
		self.opt = opt

		self.embeddings = WordVecLookup(opt)
		#self.char_embeddings = nn.Embedding(opt.char_size, opt.char_emb_size)

		# 4 stages
		self.encoder = Encoder(opt, shared)
		self.attention = BiAttention(opt, shared)
		self.reencoder = ReEncoder(opt, shared)
		self.classifier = Classifier(opt, shared)


	def init_weight(self):
		missed_names = []
		if self.opt.param_init_type == 'xavier_uniform':
			for n, p in self.named_parameters():
				if p.requires_grad:
					if 'weight' in n:
						print('initializing {}'.format(n))
						nn.init.xavier_uniform(p)
					elif 'bias' in n:
						print('initializing {}'.format(n))
						nn.init.constant(p, 0)
					else:
						missed_names.append(n)
				else:
					missed_names.append(n)
		elif self.opt.param_init_type == 'xavier_normal':
			for n, p in self.named_parameters():
				if p.requires_grad:
					if 'weight' in n:
						print('initializing {}'.format(n))
						nn.init.xavier_normal(p)
					elif 'bias' in n:
						print('initializing {}'.format(n))
						nn.init.constant(p, 0)
					else:
						missed_names.append(n)
				else:
					missed_names.append(n)
		elif self.opt.param_init_type == 'no':
			for n, p in self.named_parameters():
				missed_names.append(n)
		else:
			assert(False)

		if len(missed_names) != 0:
			print('uinitialized parameters: {0}'.format(missed_names))

		# in case needs customized initialization
		if hasattr(self.encoder, 'post_init'):
			self.encoder.post_init()
		if hasattr(self.attention, 'post_init'):
			self.attention.post_init()
		if hasattr(self.reencoder, 'post_init'):
			self.reencoder.post_init()
		if hasattr(self.classifier, 'post_init'):
			self.classifier.post_init()


	def forward(self, sent1, sent2):
		shared = self.shared

		sent1 = self.embeddings(sent1)
		sent2 = self.embeddings(sent2)
		#print(shared.batch_l, shared.context_l, shared.query_l)
		# encoder
		H, U = self.encoder(sent1, sent2)
		#sum1 = (H*H).sum() + (U*U).sum()
		# bi-attention
		att1, att2, G = self.attention(H, U)
		#sum2 = (att1*att1).sum() + (att2*att2).sum()
		# reencoder
		M = self.reencoder(G)
		#sum3 = (G*G).sum() + (M*M).sum()
		#print(sum1.data.sum(), sum2.data.sum(), sum3.data.sum())
		# classifier
		log_p1, log_p2 = self.classifier(M, G)

		return log_p1, log_p2

	# call this explicitly
	def update_context(self, batch_ex_idx, batch_l, context_l, query_l, res_map=None, raw=None):
		self.shared.batch_ex_idx = batch_ex_idx
		self.shared.batch_l = batch_l
		self.shared.context_l = context_l
		self.shared.query_l = query_l
		self.shared.res_map = res_map
		self.shared.raw = raw


	def begin_pass(self):
		self.embeddings.begin_pass()
		self.encoder.begin_pass()
		self.attention.begin_pass()
		self.reencoder.begin_pass()
		self.classifier.begin_pass()

	def end_pass(self):
		self.embeddings.end_pass()
		self.encoder.end_pass()
		self.attention.end_pass()
		self.reencoder.end_pass()
		self.classifier.end_pass()

	def get_param_dict(self):
		param_dict = {}
		if self.opt.fix_word_vecs == 0:
			param_dict.update(self.embeddings.get_param_dict('embeddings'))
		param_dict.update(self.encoder.get_param_dict('encoder'))
		param_dict.update(self.attention.get_param_dict('attention'))
		param_dict.update(self.reencoder.get_param_dict('reencoder'))
		param_dict.update(self.classifier.get_param_dict('classifier'))
		return param_dict

	def set_param_dict(self, param_dict):
		if self.opt.fix_word_vecs == 0:
			self.embeddings.set_param_dict(param_dict, 'embeddings')
		self.encoder.set_param_dict(param_dict, 'encoder')
		self.attention.set_param_dict(param_dict, 'attention')
		self.reencoder.set_param_dict(param_dict, 'reencoder')
		self.classifier.set_param_dict(param_dict, 'classifier')


def overfit():
	opt = Holder()
	opt.gpuid = 1
	opt.word_vec_size = 3
	opt.hidden_size = 6
	opt.dropout = 0.0
	opt.learning_rate = 0.05
	opt.birnn = 1
	opt.enc_rnn_layer = 1
	opt.reenc_rnn_layer = 2
	opt.cls_rnn_layer = 1
	opt.param_init_type = 'xavier_normal'
	shared = Holder()
	shared.batch_l = 2
	shared.context_l = 8
	shared.query_l = 5

	input1 = torch.randn(shared.batch_l, shared.context_l, opt.word_vec_size)
	input2 = torch.randn(shared.batch_l, shared.query_l, opt.word_vec_size)
	gold = torch.from_numpy(np.random.randint(shared.context_l, size=(shared.batch_l,2)))
	print('gold', gold)

	input1 = Variable(input1, True)
	input2 = Variable(input2, True)
	gold = Variable(gold, False)

	m = Pipeline(opt, shared)
	m.init_weight()

	crit1 = torch.nn.NLLLoss(size_average=False)
	crit2 = torch.nn.NLLLoss(size_average=False)

	# forward pass
	m.update_context(None, shared.batch_l, shared.context_l, shared.query_l, None)
	log_p1, log_p2 = m(input1, input2)
	print('p1', log_p1.exp())
	print('p2', log_p2.exp())

	

if __name__ == '__main__':
	overfit()
