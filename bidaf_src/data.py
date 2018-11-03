import h5py
import torch
from torch import nn
from torch import cuda
import numpy as np
import ujson
from util import *

class Data():
	def __init__(self, opt, data_file, res_files=None):
		self.opt = opt
		self.data_name = data_file

		print('loading data from {0}'.format(data_file))
		f = h5py.File(data_file, 'r')
		self.source = f['source'][:]
		self.target = f['target'][:]
		self.source_l = f['source_l'][:]	# (num_batch,)
		self.target_l = f['target_l'][:]	# (num_ex,)
		self.span = f['span'][:].astype(int)
		self.batch_l = f['batch_l'][:]
		self.batch_idx = f['batch_idx'][:]
		self.source_size = f['source_size'][:]
		self.target_size = f['target_size'][:]
		self.ex_idx = f['ex_idx'][:]
		self.length = self.batch_l.shape[0]
		self.seq_length = self.target.shape[1]

		self.source = torch.from_numpy(self.source)
		self.target = torch.from_numpy(self.target)
		self.span = torch.from_numpy(self.span)

		if self.opt.gpuid != -1:
			self.source = self.source.cuda()
			self.target = self.target.cuda()
			self.span = self.span.cuda()

		# load dict file
		if opt.dict != '':
			self.vocab = load_dict(opt.dict)

		self.batches = []
		for i in range(self.length):
			start = self.batch_idx[i]
			end = start + self.batch_l[i]

			# get example token indices
			source_i = self.source[start:end, 0:self.source_l[i]]
			target_i = self.target[start:end, 0:self.target_l[i]]
			span_i = self.span[start:end]

			raw = []
			if opt.dict != '':
				for k in range(end-start):
					src = source_i[k]
					tgt = target_i[k]
					src_toks = [self.vocab[idx] for idx in src]
					tgt_toks = [self.vocab[idx] for idx in tgt]
					ans = src_toks[span_i[k][0]:span_i[k][1]+1]
					raw.append((src_toks, tgt_toks, ans))

			# sanity check
			assert(self.source[start:end, self.source_l[i]:].sum() == 0)
			assert(self.target[start:end, self.target_l[i]:].sum() == 0)

			# src, tgt, batch_l, src_l, tgt_l, span, raw info
			self.batches.append((source_i, target_i, self.batch_l[i], self.source_l[i], self.target_l[i], span_i, raw))

		# count examples
		self.num_ex = 0
		for i in range(self.length):
			self.num_ex += self.batch_l[i]


		# load resource files
		self.res_names = []
		if res_files is not None:
			for f in res_files:
				print('loading res file from {0}...'.format(f))
				res_name = self.__load_json_res(f)
				# record the name
				self.res_names.append(res_name)

	def subsample(self, ratio):
		target_num_ex = int(float(self.num_ex) * ratio)
		sub_idx = torch.LongTensor(range(self.size()))

		if ratio != 1.0:
			rand_idx = torch.randperm(self.size())
			sub_num_ex = 0
			i = 0
			while sub_num_ex < target_num_ex:
				sub_num_ex += self.batch_l[rand_idx[i]]
				i += 1
			sub_idx = rand_idx[:i]

			#printing
			print('{0} examples {1} batches sampled.'.format(sub_num_ex, sub_idx.size()[0]))
		return sub_idx

	def __load_json_res(self, path):
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		# get key name of the file
		assert(len(j_obj) == 1)
		res_name = next(iter(j_obj))

		if res_name == 'content_words' or res_name == 'conj' or res_name == 'det_num':
			self.__load_json_list(path)
		else:
			self.__load_json_map(path)

		return res_name

	
	def __load_json_map(self, path):
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		# get key name of the file
		assert(len(j_obj) == 1)
		res_name = next(iter(j_obj))

		# optimize indices
		#	example id starts from 1 NOT 0 in resource file,
		#	so, shift it by -1
		res = {}
		for k, v in j_obj[res_name].iteritems():
			lut = {}
			for i, j in v.iteritems():
				if i == res_name:
					lut[res_name] = [int(l) for l in j]
				else:
					# for token indices, shift by 1 to incorporate the nul-token at the beginning
					lut[int(i)] = ([l+1 for l in j[0]], [l+1 for l in j[1]])

			# the example index starts from 1 NOT 0 in the resource file
			#	so, shift it by -1
			res[int(k)-1] = lut
		
		setattr(self, res_name, res)
		return res_name


	def __load_json_list(self, path):
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		# get key name of the file
		assert(len(j_obj) == 1)
		res_name = next(iter(j_obj))

		# optimize indices
		#	example id starts from 1 NOT 0 in resource file,
		#	so, shift it by -1
		res = {}
		for k, v in j_obj[res_name].iteritems():
			p = v['p']
			h = v['h']

			# the example index starts from 1 NOT 0 in the resource file
			#	so, shift it by -1
			# for token indices, shift by 1 to incorporate the nul-token at the beginning
			res[int(k)-1] = ([l+1 for l in p], [l+1 for l in h])
		
		setattr(self, res_name, res)
		return res_name


	def size(self):
		return self.length


	def __getitem__(self, idx):
		source, target, batch_l, source_l, target_l, span, raw = self.batches[idx]

		# get batch ex indices
		batch_ex_idx = [self.ex_idx[i] for i in range(self.batch_idx[idx], self.batch_idx[idx] + self.batch_l[idx])]

		res_map = self.__get_res(idx)

		return (self.data_name, source, target, batch_ex_idx, batch_l, source_l,  target_l, span, res_map, raw)


	def __get_res(self, idx):
		# if there is no resource presents, return None
		if len(self.res_names) == 0:
			return None


		batch_ex_idx = [self.ex_idx[i] for i in range(self.batch_idx[idx], self.batch_idx[idx] + self.batch_l[idx])]

		all_res = {}
		for res_n in self.res_names:
			res = getattr(self, res_n)

			batch_res = [res[ex_id] for ex_id in batch_ex_idx]
			all_res[res_n] = batch_res

		return all_res



if __name__ == '__main__':
	sample_data = './data/squad-val.hdf5'
	from holder import *
	opt = Holder()
	opt.gpuid = -1

	d = Data(opt, sample_data, res_files=None)
	name, src, tgt, batch_ex_idx, batch_l, src_l, tgt_l, span, res = d[113]
	print('data size: {0}'.format(d.size()))
	print('name: {0}'.format(name))
	print('source: {0}'.format(src))
	print('target: {0}'.format(tgt))
	print('batch_ex_idx: {0}'.format(batch_ex_idx))
	print('batch_l: {0}'.format(batch_l))
	print('src_l: {0}'.format(src_l))
	print('tgt_l: {0}'.format(tgt_l))
	print('span: {0}'.format(span))

	print(d.source_size)
	print(d.target_size)