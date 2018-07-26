# implementation inspired by https://github.com/Shawn1993/cnn-text-classification-pytorch

import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from join_table import *


class CharCnn(torch.nn.Module):
	def __init__(self, opt, shared):
		super(CharCnn, self).__init__()
		self.opt = opt
		self.shared = shared

		self.filter_sizes = [5]
		self.filter_dim = opt.hidden_size/2

		self.conv_layers = nn.ModuleList([nn.Conv1d(
			in_channels=opt.char_emb_size, 
			out_channels=self.filter_dim, 
			kernel_size=ngram_size) for ngram_size in self.filter_sizes])

		self.activation = nn.ReLU()

		self.post_conv_size = len(self.filter_sizes) * self.filter_dim
		self.proj = nn.Linear(self.post_conv_size, self.hidden_size/2)

		self.max_pool_joiner = JoinTable(1)
		

	# input size (batch_l, seq_l, tok_l, char_emb_size)
	def forward(self, x):
		batch_l, seq_l, tok_l, char_emb_size = x.shape
		char_emb = x.view(batch_l * seq_l, tok_l, char_emb_size).transpose(1,2)	# (batch_l * seq_l, char_emb_size, tok_l)

		max_pooled = []
		for layer in self.conv_layers:
			filtered = layer(char_emb)
			a = self.activation(filtered).max(-1)[0]	# (batch_l * seq_l, filter_dim)
			max_pooled.append(a)

		max_pooled = self.max_pool_joiner(max_pooled)	# batch_l * seq_l, filter_dim * len(filter_sizes)

		out = self.proj(max_pooled)
		out = out.view(batch_l, seq_l, self.post_conv_size)

		return out


	def get_param_dict(self, root):
		is_cuda = self.opt.gpuid != -1
		param_dict = {}
		for n, p in self.conv_layers.named_parameters():
			param_dict['{0}.conv_layers.{1}'.format(root, n)] = torch2np(p.data, is_cuda)

		param_dict['{0}.proj.weight'.format(root)] = torch2np(self.proj.weight.data, is_cuda)
		if self.proj.bias is not None:
			param_dict['{0}.proj.bias'.format(root)] = torch2np(self.proj.bias.data, is_cuda)
		
		return param_dict


	def set_param_dict(self, param_dict, root):
		for n, p in self.conv_layers.named_parameters():
			p.data.copy_(torch.from_numpy(param_dict['{0}.conv_layers.{1}'.format(root, n)][:]))

		self.proj.weight.data.copy_(torch.from_numpy(param_dict['{0}.proj.weight'.format(root)][:]))
		if self.proj.bias is not None:
			self.proj.bias.data.copy_(torch.from_numpy(param_dict['{0}.proj.bias'.format(root)][:]))




if __name__ == '__main__':
	#batch_l = 2
	#num_kernel = 3
	#emb_size = 4
	#seq_l = 4
	#filter_size = 3
	#input_channel = 1
	#a = Variable(torch.ones(batch_l, seq_l, emb_size))
	#a = a.unsqueeze(1)
	#conv = nn.Conv2d(input_channel, num_kernel, (filter_size, emb_size))
#
	#print('a', a)
	#print('conv', conv)
#
	#
	#out = conv(a)
	#print('out', out)
#
	#out = out.squeeze(-1)
#
	#max_out = nn.MaxPool1d(out.size(2))(out)
	#print('max_out', max_out)



	shared = Holder()
	shared.seq_l = 4
	opt = Holder()
	opt.batch_l = 2
	opt.num_kernel = 8
	opt.char_emb_size = 4
	opt.char_cnn_kernels = '3'
	opt.dropout = 0.0

	conv = CharCNN(opt, shared)
	a = Variable(torch.ones(opt.batch_l, shared.seq_l, opt.char_emb_size))
	print('a', a)

	out = conv(a)
	print('out', out)


