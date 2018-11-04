import sys
import h5py
import torch
from torch import nn
from torch import cuda
from .view import *
from .join_table import *
from .holder import *
from .util import *


class WordVecLookup(torch.nn.Module):
	def __init__(self, opt):
		super(WordVecLookup, self).__init__()
		self.opt = opt
	
		print('loading word vector from {0}'.format(opt.word_vecs))
		f = h5py.File(opt.word_vecs, 'r')
		word_vecs = f['word_vecs'][:]
		self.embeddings = nn.Embedding(word_vecs.shape[0], word_vecs.shape[1])
		self.embeddings.weight.data.copy_(torch.from_numpy(word_vecs))

		self.embeddings.weight.requires_grad = opt.fix_word_vecs == 0

	def forward(self, idx):
		return self.embeddings(idx)

	def cuda(self):
		if self.opt.gpuid != -1:
			torch.cuda.set_device(self.opt.gpuid)
			self.embeddings = self.embeddings.cuda()


	def get_param_dict(self, root):
		is_cuda = self.opt.gpuid != -1
		param_dict = {}
		# w
		param_dict['{0}.weight'.format(root)] = torch2np(self.embeddings.weight.data, is_cuda)

		return param_dict

	def set_param_dict(self, param_dict, root):
		self.embeddings.weight.data.copy_(torch.from_numpy(param_dict['{0}.embeddings.weight'.format(root)][:]))


	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	from torch.autograd import Variable
	opt = Holder()
	opt.word_vecs = '../learnlab/data/glove.hdf5'
	opt.fix_word_vecs = 1
	wv = WordVecLookup(opt)
	
	idx = Variable(torch.LongTensor([0, 10, 5]), False)
	vecs = wv(idx)

	print(vecs)

