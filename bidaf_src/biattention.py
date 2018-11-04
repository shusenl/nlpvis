import sys

import torch
from torch import nn
from torch.autograd import Variable
from .view import *
from .holder import *
from .util import *
from .join_table import *


# bidir attention
class BiAttention(torch.nn.Module):
	def __init__(self, opt, shared):
		super(BiAttention, self).__init__()
		self.opt = opt
		self.shared = shared

		self.w_hu = nn.Sequential(
			nn.Linear(opt.hidden_size*3, 1))

		self.score_view = View(1,1,1,1)	# (batch_l, context_l, query_l, hidden_size)
		self.score_unview = View(1,1,1)	# (batch_l, context_l, query_l)

		self.softmax1 = nn.Softmax(2)
		self.softmax2 = nn.Softmax(1)
		self.g_joiner = JoinTable(2)
		self.hu_joiner = JoinTable(3)


	def biattention(self, scores1, H, U):
		batch_l = self.shared.batch_l
		context_l = self.shared.context_l
		hidden_size = self.opt.hidden_size

		# attention
		att1 = self.softmax1(scores1)			# (batch_l, context_l, query_l)
		att2 = self.softmax2(scores1.max(2)[0])	# (batch_l, context_l)
		att2 = att2.unsqueeze(1)				# (batch_l, 1, context_l)

		# attend
		agg1 = att1.bmm(U)	# (batch_l, context_l, hidden_size)
		agg2 = att2.bmm(H)	# (batch_l, 1, hidden_size)
		agg2 = agg2.expand(batch_l, context_l, hidden_size)
		G = self.g_joiner([H, agg1, H * agg1, H * agg2])
		return [att1, att2, G]


	# input encodings of context (H) and query (U)
	#	H of shape (batch_l, context_l, hidden_size)
	#	U of shape (batch_l, query_l, hidden_size)
	def forward(self, H, U):
		self.update_context()
		batch_l = self.shared.batch_l
		context_l = self.shared.context_l
		query_l = self.shared.query_l
		hidden_size = self.opt.hidden_size

		# upscaling
		H_up = H.unsqueeze(2).expand(batch_l, context_l, query_l, hidden_size)
		U_up = U.unsqueeze(1).expand(batch_l, context_l, query_l, hidden_size)
		HU_phi = self.hu_joiner([H_up, U_up, H_up * U_up])	# (batch_l, context_l, query_l, hidden_size * 3)

		# get similarity score
		scores1 = self.score_unview(self.w_hu(self.score_view(HU_phi)))	# (batch_l, context_l, query_l)

		#
		att1, att2, G = self.biattention(scores1, H, U)

		# bookkeeping
		self.shared.score1 = scores1
		self.shared.att_soft1 = att1
		self.shared.att_soft2 = att2
		self.shared.G = G

		return [att1, att2, G]


	def update_context(self):
		batch_l = self.shared.batch_l
		context_l = self.shared.context_l
		query_l = self.shared.query_l
		word_vec_size = self.opt.word_vec_size
		hidden_size = self.opt.hidden_size

		self.score_view = View(batch_l * context_l * query_l, hidden_size * 3)
		self.score_unview = View(batch_l, context_l, query_l)


	def get_param_dict(self, root):
		is_cuda = self.opt.gpuid != -1
		param_dict = {}
		# w
		for i in [0]:
			param_dict['{0}.w_hu.{1}.weight'.format(root, i)] = torch2np(self.w_hu[i].weight.data, is_cuda)
			if self.w_hu[i].bias is not None:
				param_dict['{0}.w_hu.{1}.bias'.format(root, i)] = torch2np(self.w_hu[i].bias.data, is_cuda)

		return param_dict

	def set_param_dict(self, param_dict, root):
		for i in [0]:
			self.w_hu[i].weight.data.copy_(torch.from_numpy(param_dict['{0}.w_hu.{1}.weight'.format(root, i)][:]))
			if self.w_hu[i].bias is not None:
				self.w_hu[i].bias.data.copy_(torch.from_numpy(param_dict['{0}.w_hu.{1}.bias'.format(root, i)][:]))


	def begin_pass(self):
		pass

	def end_pass(self):
		pass
