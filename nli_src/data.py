import h5py
import torch
from torch import nn
from torch import cuda
import numpy as np

class Data():
	def __init__(self, opt, data_file, res_list=None):
		self.opt = opt
		self.data_name = data_file

		print('loading data from {0}'.format(data_file))
		f = h5py.File(data_file, 'r')
		self.source = f['source'][:]
		self.target = f['target'][:]
		self.source_l = f['source_l'][:]
		self.target_l = f['target_l'][:]
		self.label = f['label'][:]
		self.batch_l = f['batch_l'][:]
		self.batch_idx = f['batch_idx'][:]
		self.source_size = f['source_size'][:]
		self.target_size = f['target_size'][:]
		self.label_size = f['label_size'][:]
		self.ex_idx = f['ex_idx'][:]
		self.length = self.batch_l.shape[0]
		self.seq_length = self.target.shape[1]

		self.source = torch.from_numpy(self.source)
		self.target = torch.from_numpy(self.target)
		self.label = torch.from_numpy(self.label)

		if self.opt.gpuid != -1:
			self.source = self.source.cuda()
			self.target = self.target.cuda()
			self.label = self.label.cuda()

		# shift -1 to start from 0
		self.batch_idx -= 1
		self.source -= 1
		self.target -= 1
		self.label -= 1
		self.ex_idx -= 1

		self.batches = []
		for i in range(self.length):
			start = self.batch_idx[i]
			end = start + self.batch_l[i]
			source_i = self.source[start:end, 0:self.source_l[i]]
			target_i = self.target[start:end, 0:self.target_l[i]]
			label_i = self.label[start:end]

			# src, tgt, batch_l, src input size, tgt input size, label
			self.batches.append((source_i, target_i, self.batch_l[i], self.source_l[i], self.target_l[i], label_i))

		# load resource files

		#

	def size(self):
		return self.length

	def __getitem__(self, idx):
		source, target, batch_l, source_l, target_l, label = self.batches[idx]

		# get batch ex indices
		batch_ex_idx = [self.ex_idx[i] for i in range(self.batch_idx[idx], self.batch_idx[idx] + self.batch_l[idx])]

		return (self.data_name, source, target, batch_ex_idx, batch_l, source_l, target_l, label)


if __name__ == '__main__':
	sample_data = '../data/snli_1.0-val.hdf5'
	from holder import *
	opt = Holder()
	opt.gpuid = -1

	d = Data(opt, sample_data)
	name, src, tgt, batch_ex_idx, batch_l, src_l, tgt_l, label = d[100]
	print('data size: {0}'.format(d.size()))
	print('name: {0}'.format(name))
	print('source: {0}'.format(src))
	print('target: {0}'.format(tgt))
	print('batch_ex_idx: {0}'.format(batch_ex_idx))
	print('batch_l: {0}'.format(batch_l))
	print('src_l: {0}'.format(src_l))
	print('tgt_l: {0}'.format(tgt_l))
	print('label: {0}'.format(label))

	print(d.source_size)
	print(d.target_size)
