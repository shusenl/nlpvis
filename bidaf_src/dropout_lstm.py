import sys

import torch
from torch import nn
from torch.autograd import Variable
from view import *
from holder import *
from util import *
from join_table import *


# customized lstm with inter-timestep dropout
# 	dropout mask is consistent across time steps
# batch-first is assumed
class DropoutLSTMLayer(torch.nn.Module):
	def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, bidirectional=False):
		super(DropoutLSTMLayer, self).__init__()

		is_cuda = torch.cuda.is_available()
		self.drop = dropout
		self.bidirectional = bidirectional
		#self.output_hidden_size = hidden_size if bidirectional is False else hidden_size*2

		#self.drop_fw = nn.Dropout(dropout)
		self.cell_fw = nn.LSTMCell(input_size, hidden_size, bias)
		self.fw_joiner = JoinTable(1)
		self.h_fw_init = Variable(torch.zeros(1,self.cell_fw.hidden_size), requires_grad=False)
		self.c_fw_init = Variable(torch.zeros(1,self.cell_fw.hidden_size), requires_grad=False)
		if is_cuda:
			self.h_fw_init = self.h_fw_init.cuda()
			self.c_fw_init = self.c_fw_init.cuda()
			
		if bidirectional:
			#self.drop_bw = nn.Dropout(dropout)
			self.cell_bw = nn.LSTMCell(input_size, hidden_size, bias)	
			self.bw_joiner = JoinTable(1)
			self.fw_bw_joiner = JoinTable(2)
			self.h_bw_init = Variable(torch.zeros(1, self.cell_fw.hidden_size), requires_grad=False)
			self.c_bw_init = Variable(torch.zeros(1, self.cell_bw.hidden_size), requires_grad=False)
			if is_cuda:
				self.h_bw_init = self.h_bw_init.cuda()
				self.c_bw_init = self.c_bw_init.cuda()
			


	# change the dropout ratio
	def dropout(self, p):
		self.drop = p
		#self.drop_fw.p = p
		#if self.bidirectional:
		#	self.drop_bw.p = p


	def sample_dropout(self):
		is_cuda = torch.cuda.is_available()
		scale = 1.0 / (1.0 - self.drop)
		self.fw_x_mask = Variable(
			torch.bernoulli(torch.Tensor(1, 1, self.cell_fw.input_size).fill_(1.0 - self.drop)) * scale,
			requires_grad=False)
		self.fw_h_mask = Variable(
			torch.bernoulli(torch.Tensor(1, self.cell_fw.hidden_size).fill_(1.0 - self.drop)) * scale,
			requires_grad=False)
		self.bw_x_mask = Variable(
			torch.bernoulli(torch.Tensor(1, 1, self.cell_bw.input_size).fill_(1.0 - self.drop)) * scale,
			requires_grad=False)
		self.bw_h_mask = Variable(
			torch.bernoulli(torch.Tensor(1, self.cell_bw.hidden_size).fill_(1.0 - self.drop)) * scale,
			requires_grad=False)
		if is_cuda:
			self.fw_x_mask = self.fw_x_mask.cuda()
			self.fw_h_mask = self.fw_h_mask.cuda()
			self.bw_x_mask = self.bw_x_mask.cuda()
			self.bw_h_mask = self.bw_h_mask.cuda()


	# x must be of shape (batch_l, seq_l, input_size)
	# For simplicity, will output no hidden states
	def forward(self, x):
		self.sample_dropout()

		batch_l, seq_l, input_size = x.shape

		# forward pass
		output_fw = [None for _ in xrange(seq_l)]
		h_fw = self.h_fw_init.expand(batch_l, -1)
		c_fw = self.c_fw_init.expand(batch_l, -1)
		x_fw = x * self.fw_x_mask
		for i in xrange(seq_l):
			h_fw1 = h_fw * self.fw_h_mask
			h_fw, c_fw = self.cell_fw(x[:, i, :], (h_fw, c_fw))
			output_fw[i] = h_fw.unsqueeze(1)	# (batch_l, 1, hidden_size)

		rs = self.fw_joiner(output_fw)

		if self.bidirectional is True:
			output_bw = [None for _ in xrange(seq_l)]
			h_bw = self.h_bw_init.expand(batch_l, -1)
			c_bw = self.c_bw_init.expand(batch_l, -1)
			x_bw = x * self.bw_x_mask
			for i in xrange(seq_l-1, -1, -1):
				h_bw1 = h_bw * self.bw_h_mask
				h_bw, c_bw = self.cell_bw(x[:, i, :], (h_bw1, c_bw))
				output_bw[i] = h_bw.unsqueeze(1)	# (batch_l, 1, hidden_size)

			rs = self.fw_bw_joiner([rs, self.bw_joiner(output_bw)])

		return rs, None	# output no hidden states


class DropoutLSTM(torch.nn.Module):
	def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0.0, bidirectional=False):
		super(DropoutLSTM, self).__init__()

		assert(batch_first is True)

		self.rnn_layers = []
		for i in xrange(num_layers):
			in_size = input_size
			if i > 0:
				if bidirectional:
					in_size = hidden_size*2
				else:
					in_size = hidden_size

			self.rnn_layers.append(DropoutLSTMLayer(in_size, hidden_size, bias, dropout, bidirectional))

		self.rnn_layers = nn.ModuleList(self.rnn_layers)
		self.num_layers = num_layers


	# change the dropout ratio
	def dropout(self, p):
		for i in xrange(self.num_layers):
			self.rnn_layers[i].dropout(p)


	# x must be of shape (batch_l, seq_l, input_size)
	# For simplicity, will output no hidden states
	def forward(self, x):
		for i in xrange(self.num_layers):
			x, _ = self.rnn_layers[i](x)

		return x, None



if __name__ == '__main__':
	layer = DropoutLSTMLayer(10, 3, bidirectional=True)
	for n,p in layer.named_parameters():
		print(n)
	x = Variable(torch.ones(2, 4, 10))
	y, _ = layer(x)
	print(y)



	lstm = DropoutLSTM(10, 3, num_layers=2, bidirectional=True)
	y, _ = lstm(x)
	print(y)
		