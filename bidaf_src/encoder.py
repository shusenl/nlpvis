import sys

import torch
from torch import nn
from torch.autograd import Variable
from view import *
from holder import *
from util import *
from highway import *
from join_table import *
from dropout_lstm import *
from locked_dropout import *

# encoder
class Encoder(torch.nn.Module):
	def __init__(self, opt, shared):
		super(Encoder, self).__init__()
		self.opt = opt
		self.shared = shared

		# sampling to hidden_size embeddings
		self.sampler = nn.Linear(opt.word_vec_size, opt.hidden_size)

		# highway
		self.highway = Highway(opt, shared)

		# viewer
		self.context_view = View(1,1)
		self.context_unview = View(1,1,1)
		self.query_view = View(1,1)
		self.query_unview = View(1,1,1)

		# dropout for rnn
		self.drop1 = nn.Dropout(opt.dropout)

		# rnn after highway
		self.bidir = opt.birnn == 1
		rnn_in_size = opt.hidden_size	# input size is the output size of highway
		rnn_hidden_size = opt.hidden_size if not self.bidir else opt.hidden_size/2
		if opt.rnn_type == 'dropout_lstm':
			# use customized lstm with inter-timestep dropout
			self.rnn = DropoutLSTM(
				input_size=rnn_in_size, 
				hidden_size=rnn_hidden_size, 
				num_layers=opt.enc_rnn_layer,
				bias=True,
				batch_first=True,
				dropout=opt.dropout_h,
				bidirectional=self.bidir)

		elif opt.rnn_type == 'lstm':
			self.rnn = nn.LSTM(
				input_size=rnn_in_size,
				hidden_size=rnn_hidden_size, 
				num_layers=opt.enc_rnn_layer,
				bias=True,
				batch_first=True,
				dropout=opt.dropout_h,
				bidirectional=self.bidir)

		elif opt.rnn_type == 'gru':
			self.rnn = nn.GRU(
				input_size=rnn_in_size,
				hidden_size=rnn_hidden_size, 
				num_layers=opt.enc_rnn_layer,
				bias=True,
				batch_first=True,
				dropout=opt.dropout_h,
				bidirectional=self.bidir)

		else:
			assert(False)

		self.rnn_joiner = JoinTable(1)


	def rnn_over(self, seq):
		if self.opt.rnn_type == 'dropout_lstm':
			E, _ = self.rnn(seq)
			return E

		elif self.opt.rnn_type == 'lstm' or self.opt.rnn_type == 'gru':
			E, _ = self.rnn(seq)
			E = self.drop1(E)
			return E

		else:
			assert(False)


	# context of shape (batch_l, context_l, word_vec_size)
	# query of shape (batch_l, query_l, word_vec_size)
	def forward(self, context, query):
		self.update_context()

		# sampling
		H = self.sampler(self.context_view(context))
		U = self.sampler(self.query_view(query))

		# highway
		# context will be (batch_l, context_l, word_vec_size)
		# query will be (batch_l, query_l, word_vec_size)
		context = self.context_unview(self.highway(H))
		query = self.query_unview(self.highway(U))

		# rnn
		H = self.rnn_over(context)
		U = self.rnn_over(query)

		self.shared.H = H
		self.shared.U = U

		# sanity check
		assert(H.shape == (self.shared.batch_l, self.shared.context_l, self.opt.hidden_size))
		assert(U.shape == (self.shared.batch_l, self.shared.query_l, self.opt.hidden_size))

		return [self.shared.H, self.shared.U]


	def update_context(self):
		batch_l = self.shared.batch_l
		context_l = self.shared.context_l
		query_l = self.shared.query_l
		hidden_size = self.opt.hidden_size

		self.context_view.dims = (batch_l * context_l, self.opt.word_vec_size)
		self.context_unview.dims = (batch_l, context_l, hidden_size)
		self.query_view.dims = (batch_l * query_l, self.opt.word_vec_size)
		self.query_unview.dims = (batch_l, query_l, hidden_size)


	def get_param_dict(self, root):
		is_cuda = self.opt.gpuid != -1
		param_dict = {}
		# sample
		param_dict['{0}.sampler.weight'.format(root)] = torch2np(self.sampler.weight.data, is_cuda)
		if self.sampler.bias is not None:
			param_dict['{0}.sampler.bias'.format(root)] = torch2np(self.sampler.bias.data, is_cuda)
		# hgihway
		param_dict.update(self.highway.get_param_dict(root + '.highway'))
		# rnn layer
		for n, p in self.rnn.named_parameters():
			param_dict['{0}.rnn.{1}'.format(root, n)] = torch2np(p.data, is_cuda)

		return param_dict


	def set_param_dict(self, param_dict, root):
		# sampler
		self.sampler.weight.data.copy_(torch.from_numpy(param_dict['{0}.sampler.weight'.format(root)][:]))
		if self.sampler.bias is not None:
			self.sampler.bias.data.copy_(torch.from_numpy(param_dict['{0}.sampler.bias'.format(root)][:]))
		# highway
		self.highway.set_param_dict(param_dict, root + '.highway')
		# rnn layer
		for n, p in self.rnn.named_parameters():
			p.data.copy_(torch.from_numpy(param_dict['{0}.rnn.{1}'.format(root, n)][:]))


	def post_init(self):
		if hasattr(self.highway, 'post_init'):
			self.highway.post_init()

	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	pass





	
