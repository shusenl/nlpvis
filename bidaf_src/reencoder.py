import sys

import torch
from torch import nn
from torch.autograd import Variable
from .view import *
from .holder import *
from .util import *
from .join_table import *
from .dropout_lstm import *
from .locked_dropout import *

# re-encoder
class ReEncoder(torch.nn.Module):
	def __init__(self, opt, shared):
		super(ReEncoder, self).__init__()
		self.opt = opt
		self.shared = shared

		# dropout for rnn
		self.drop1 = nn.Dropout(opt.dropout)

		self.bidir = opt.birnn == 1
		rnn_in_size = opt.hidden_size * 4
		rnn_hidden_size = opt.hidden_size if not self.bidir else opt.hidden_size/2
		
		if opt.rnn_type == 'dropout_lstm':
			# use customized lstm with inter-timestep dropout
			self.rnn = DropoutLSTM(
				input_size=rnn_in_size,
				hidden_size=rnn_hidden_size, 
				num_layers=opt.reenc_rnn_layer,
				bias=True,
				batch_first=True,
				dropout=opt.dropout_h,
				bidirectional=self.bidir)

		elif opt.rnn_type == 'lstm':
			self.rnn = nn.LSTM(
				input_size=rnn_in_size,
				hidden_size=rnn_hidden_size, 
				num_layers=opt.reenc_rnn_layer,
				bias=True,
				batch_first=True,
				dropout=opt.dropout_h,
				bidirectional=self.bidir)

		elif opt.rnn_type == 'gru':
			self.rnn = nn.GRU(
				input_size=rnn_in_size,
				hidden_size=rnn_hidden_size, 
				num_layers=opt.reenc_rnn_layer,
				bias=True,
				batch_first=True,
				dropout=opt.dropout_h,
				bidirectional=self.bidir)

		else:
			assert(False)


	def rnn_over(self, context):
		if self.opt.rnn_type == 'dropout_lstm':
			M, _ = self.rnn(context)
			return M
		elif self.opt.rnn_type == 'lstm' or self.opt.rnn_type == 'gru':
			M, _ = self.rnn(context)
			M = self.drop1(M)
			return M
		else:
			assert(False)


	def forward(self, G):
		self.update_context()

		M = self.rnn_over(G)

		self.shared.M = M

		return M


	def update_context(self):
		pass


	def get_param_dict(self, root):
		is_cuda = self.opt.gpuid != -1
		param_dict = {}
		# rnn layer
		for n, p in self.rnn.named_parameters():
			param_dict['{0}.rnn.{1}'.format(root, n)] = torch2np(p.data, is_cuda)

		return param_dict

	def set_param_dict(self, param_dict, root):
		# rnn layer
		for n, p in self.rnn.named_parameters():
			p.data.copy_(torch.from_numpy(param_dict['{0}.rnn.{1}'.format(root, n)][:]))

	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	pass





	
