# defines some callback functions registered in pipeline
#	registered functions will be called during forward pass
#	for instance, add one of the functions to pipeline during eval.py running
import torch
from torch import nn
from torch import cuda
from holder import *
import numpy as np
import h5py

# hacky flag
def is_first_time(shared):
	if '__first_time_mark' not in shared:
		shared.__first_time_mark = 1
		return True
	return False

# print out attention to hdf5 file
# NOTE, to make this work, the desired attention must be registed in the shared holder
#	e.g. shared.att_soft1 or so
# att_name specifies the variable name of the attention, "att_soft1" in this case
def print_attention(opt, shared, m, att_name):
	
	outpath = '{0}.hdf5'.format(opt.att_output)

	# if it's first time being called, clear up the file
	if is_first_time(shared):
		print('touching {0}'.format(outpath))
		file = h5py.File(outpath, 'w')
		file.close()
	# open it in append mode
	file = h5py.File(outpath, 'a')

	batch_att = getattr(shared, att_name)
	print('printing {0} for {1} examples...'.format(att_name, shared.batch_l))
	for i in xrange(shared.batch_l):
		ex_id = shared.batch_ex_idx[i]
		att = batch_att.data[i, :, :]

		# in case there are multiple att labels
		#	write 3d tensor
		# otherwise
		#	write 2d tensor
		if opt.num_att_labels != 1:
			chunks = []
			sent_l2 = att.shape[2] / self.num_att_labels
			for l in xrange(opt.num_att_labels):
				# get the chunk and transpose it
				chunks.append(att[:, l*sent_l2:(l+1)*sent_l2].unsqueeze(0))
			att_3d = torch.cat(chunks, 0)

			file.create_dataset('{0}'.format(ex_id), data=att_3d)
		else:
			file.create_dataset('{0}'.format(ex_id), data=att)

	file.close()

def print_score1(opt, shared, m):
	print_attention(opt, shared, m, 'score1')

def print_score2(opt, shared, m):
	print_attention(opt, shared, m, 'score2')

def print_att_soft1(opt, shared, m):
	print_attention(opt, shared, m, 'att_soft1')

def print_att_soft2(opt, shared, m):
	print_attention(opt, shared, m, 'att_soft2')

# this register all hooks defined
#	hooks are called according to opt.forward_hooks (a list of function names)
def run_forward_hooks(opt, shared, m):
	if opt.forward_hooks == '':
		return

		
	hooks = {}
	hooks['print_score1'] = print_score1
	hooks['print_score2'] = print_score2
	hooks['print_att_soft1'] = print_att_soft1
	hooks['print_att_soft2'] = print_att_soft2

	names = opt.forward_hooks.split(',')
	for name in names:
		cb = hooks[name]
		cb(opt, shared, m)


