import sys
import h5py
import torch
from torch import cuda


def torch2np(t, is_cuda):
	return t.numpy() if not is_cuda else t.cpu().numpy()


def save_opt(opt, path):
	with open(path, 'w') as f:
		f.write('{0}'.format(opt))


def load_param_dict(path):
	# TODO, this is ugly
	f = h5py.File(path, 'r')
	return f


def save_param_dict(param_dict, path):
	file = h5py.File(path, 'w')
	for name, p in param_dict.iteritems():
		file.create_dataset(name, data=p)

	file.close()


def load_dict(path):
	rs = {}
	with open(path, 'r+') as f:
		for l in f:
			if l.strip() == '':
				continue
			w, idx = l.strip().split()
			rs[int(idx)] = w
	return rs