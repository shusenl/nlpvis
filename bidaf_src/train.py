import sys
from .pipeline import *
import argparse
import h5py
import os
import random
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch import cuda
from .holder import *
from .optimizer import *
from .embeddings import *
from .data import *
from .util import *
from .ema import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--train_data', help="Path to training data hdf5 file.", default="data/squad-train.hdf5")
parser.add_argument('--val_data', help="Path to validation data hdf5 file.", default="data/squad-val.hdf5")
parser.add_argument('--save_file', help="Path to where model to be saved.", default="model")
parser.add_argument('--word_vecs', help="The path to word embeddings", default = "")
parser.add_argument('--dict', help="The path to word dictionary", default = "")
# resource specs
parser.add_argument('--train_res', help="Path to training resource files, seperated by comma.", default="")
parser.add_argument('--val_res', help="Path to validation resource files, seperated by comma.", default="")
## pipeline specs
parser.add_argument('--word_vec_size', help="The input word embedding dim", type=int, default=100)
parser.add_argument('--hidden_size', help="The general hidden size of the pipeline, twice of BIDAF's d", type=int, default=200)
parser.add_argument('--dropout', help="The dropout probability", type=float, default=0.2)
parser.add_argument('--dropout_h', help="The dropout probability on rnn hidden states", type=float, default=0.2)
parser.add_argument('--percent', help="The percent of training data to use", type=float, default=1.0)
parser.add_argument('--epochs', help="The number of epoches for training", type=int, default=500)
parser.add_argument('--optim', help="The name of optimizer to use for training", default='adagrad')
parser.add_argument('--mu', help="The mu ratio used in EMA", type=float, default=0.999)
# TODO, param_init of uniform dist or normal dist???
parser.add_argument('--param_init_type', help="The type of parameter initialization", default='xavier_uniform')
parser.add_argument('--learning_rate', help="The learning rate for training", type=float, default=0.5)
parser.add_argument('--fix_word_vecs', help="Whether to make word embeddings NOT learnable", type=int, default=1)
parser.add_argument('--print_every', help="Print stats after this many batches", type=int, default=1000)
parser.add_argument('--seed', help="The random seed", type=int, default=3435)
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--clip', help="The norm2 threshold to clip, set it to negative to disable", type=float, default=-1.0)
parser.add_argument('--enc_rnn_layer', help="The number of layers of rnn encoder", type=int, default=1)
parser.add_argument('--reenc_rnn_layer', help="The number of layers of rnn reencoder", type=int, default=2)
parser.add_argument('--cls_rnn_layer', help="The number of layers of classifier rnn", type=int, default=1)
parser.add_argument('--birnn', help="Whether to use bidirectional rnn", type=int, default=1)
parser.add_argument('--rnn_type', help="The type of rnn to use (lstm or gru)", default='lstm')
parser.add_argument('--hw_layer', help="The number of highway layers to use", type=int, default=2)
parser.add_argument('--ema', help="Whether to use EMA", type=int, default=0)
parser.add_argument('--grad_cache_size', help="The number of batches to delay update", type=int, default=1)

# dist: torch tensor of distribution (batch_l, context_l)
def pick_span(p1, p2):
	start = p1.max(1)[1].unsqueeze(-1)
	end = p2.max(1)[1].unsqueeze(-1)
	return torch.cat([start, end], 1)


# pick the i,j that i <= j
# input torch tensors of distribution (batch_l, context_l)
def pick_constrained_span(p1, p2, is_cuda=False):
	# product of probabilities in 2d for each example
	mats = p1.unsqueeze(-1) * p2.unsqueeze(1)	# (batch_l, context_l, context_l)
	#
	spans = []
	for i in range(mats.shape[0]):
		# get the upper triangular matrix
		triu = np.triu(mats[i].cpu().numpy())
		# get the max index
		max_idx = np.argmax(triu)
		max_idx = np.unravel_index(max_idx, triu.shape)
		spans.append([max_idx[0], max_idx[1]])

	spans = torch.Tensor(spans).long()	# (batch_l, 2)
	if is_cuda:
		spans = spans.cuda()
	return spans

def pick_idx(p):
	p = p.cpu().numpy()
	return np.argmax(p, axis=1)

def count_correct_idx(pred, gold):
	return np.equal(pred, gold).sum()


# pred: torch tensor of shape (batch_l, 2)
# gold: torch tensor of shape (batch_l, 2)
def get_span_f1(pred, gold):
	pred = pred.cpu()
	gold = gold.cpu()
	gold_start = gold[:,0]
	gold_end = gold[:,1] + 1	# exclusive ending
	pred_start = pred[:,0]
	pred_end = pred[:,1] + 1	# exclusive ending

	start = torch.max(pred_start, gold_start)
	end = torch.min(pred_end, gold_end)

	pred_range = pred_end - pred_start
	gold_range = gold_end - gold_start
	overlap = (end - start).clamp(min=0)

	# recall
	rec = overlap.float() / gold_range.float()

	# numerical fixes for precision
	pred_range = pred_range.clamp(min=0)
	denom = pred_range.float()
	denom_mask = (pred_range == 0).float()
	prec = overlap.float() / (denom + denom_mask)
	
	# numerical fixes for f1
	denom = prec + rec
	denom_mask = (denom == 0.0).float()
	f1 = 2.0 * prec * rec / (denom + denom_mask)

	return (prec, rec, f1)

def train_batch(opt, shared, m, optim, ema, data, epoch_id, sub_idx=None):
	train_loss = 0.0
	num_ex = 0
	start_time = time.time()
	train_idx1_correct = 0
	train_idx2_correct = 0
	min_grad_norm2 = 1000000000000.0
	max_grad_norm2 = 0.0

	# subsamples of data
	# if subsample indices provided, permutate from subsamples
	#	else permutate from all the data
	data_size = data.size() if sub_idx is None else sub_idx.size()[0]
	batch_order = torch.randperm(data_size)
	if sub_idx is not None:
		batch_order = sub_idx[batch_order]

	m.train(True)
	m.begin_pass()
	for i in range(data_size):
		data_name, source, target, batch_ex_idx, batch_l, source_l, target_l, span, res_map, raw = data[batch_order[i]]

		wv_idx1 = Variable(source, requires_grad=False)
		wv_idx2 = Variable(target, requires_grad=False)
		y_gold = Variable(span, requires_grad=False)

		# update network parameters
		shared.epoch = epoch_id
		m.update_context(batch_ex_idx, batch_l, source_l, target_l, res_map, raw)

		# forward pass
		log_p1, log_p2 = m.forward(wv_idx1, wv_idx2)

		# loss
		crit1 = torch.nn.NLLLoss(size_average=False)
		crit2 = torch.nn.NLLLoss(size_average=False)
		if opt.gpuid != -1:
			crit1 = crit1.cuda()
			crit2 = crit2.cuda()
		loss1 = crit1(log_p1, y_gold[:,0])	# loss on start idx
		loss2 = crit2(log_p2, y_gold[:,1])	# loss on end idx
		loss = (loss1 + loss2) / batch_l

		# stats
		idx1 = pick_idx(log_p1.data)
		idx2 = pick_idx(log_p2.data)
		train_idx1_correct += count_correct_idx(idx1, y_gold[:,0].data)
		train_idx2_correct += count_correct_idx(idx2, y_gold[:,1].data)

		# backward pass
		m.zero_grad()
		loss.backward()

		# update network weights
		grad_norm2 = optim.step(m)
		if opt.ema == 1:
			ema.step(m)

		# printing
		grad_norm2_avg = grad_norm2 / batch_l
		min_grad_norm2 = min(min_grad_norm2, grad_norm2_avg)
		max_grad_norm2 = max(max_grad_norm2, grad_norm2_avg)
		train_loss += loss.data[0] * batch_l
		num_ex += batch_l
		time_taken = time.time() - start_time

		if (i+1) % opt.print_every == 0:
			stats = '{0}, Batch {1:.1f}k '.format(epoch_id+1, float(i+1)/1000)
			stats += 'Grad {0:.1f}/{1:.1f} '.format(min_grad_norm2, max_grad_norm2)
			stats += 'Loss {0:.3f} EM {1:.3f}/{2:.3f} '.format(
				train_loss / num_ex, 
				float(train_idx1_correct) / num_ex,
				float(train_idx2_correct) / num_ex)
			stats += 'Time {0:.1f}'.format(time_taken)
			print(stats)
	m.end_pass()

	acc1 = float(train_idx1_correct) / num_ex
	acc2 = float(train_idx2_correct) / num_ex

	return train_loss / num_ex, num_ex, acc1, acc2

def train(opt, shared, m, optim, ema, train_data, valid_data):
	best_val_perf = 0.0
	test_perf = 0.0
	train_perfs = []
	val_perfs = []

	sub_idx = train_data.subsample(opt.percent) if opt.percent != 1.0 else None
	start = 0

	for i in range(start, opt.epochs):
		loss, num_ex, train_acc1, train_acc2 = train_batch(opt, shared, m, optim, ema, train_data, i, sub_idx)
		train_acc = (train_acc1 + train_acc2) / 2.0
		train_perfs.append(train_acc)
		print('Train {0:.4f}/{1:.4f}, avg {2:.4f}'.format(train_acc1, train_acc2, train_acc))

		# evaluate
		#	and save if it's the best model
		val_acc1, val_acc2, val_loss = validate(opt, shared, m, valid_data)
		val_acc = (val_acc1 + val_acc2)/2.0
		val_perfs.append(val_acc)
		print('Val {0:.4f}/{1:.4f}, avg {2:.4f}'.format(val_acc1, val_acc2, val_acc))

		str_perf_table = ''
		for i in range(len(train_perfs)):
			str_perf_table += '{0}\t{1:.6f}\t{2:.6f}\n'.format(i+1, train_perfs[i], val_perfs[i])
		print(str_perf_table)

		if val_acc > best_val_perf:
			best_val_perf = val_acc
			print('saving model to {0}'.format(opt.save_file))
			param_dict = m.get_param_dict()
			save_param_dict(param_dict, '{0}.hdf5'.format(opt.save_file))
			save_opt(opt, '{0}.opt'.format(opt.save_file))
			# save ema
			if opt.ema == 1:
				ema_param_dict = ema.get_param_dict()
				save_param_dict(ema_param_dict, '{0}.ema.hdf5'.format(opt.save_file))
			
		else:
			print('skip saving model for perf <= {0:.4f}'.format(best_val_perf))



def validate(opt, shared, m, data):
	m.train(False)

	total_loss = 0.0
	num_ex = 0
	val_idx1_correct = 0
	val_idx2_correct = 0

	m.begin_pass()
	for i in range(data.size()):
		data_name, source, target, batch_ex_idx, batch_l, source_l, target_l, label, res_map, raw = data[i]

		wv_idx1 = Variable(source, requires_grad=False)
		wv_idx2 = Variable(target, requires_grad=False)
		y_gold = Variable(label, requires_grad=False)
		# set resources, TODO

		# update network parameters
		m.update_context(batch_ex_idx, batch_l, source_l, target_l, res_map, raw)

		# forward pass
		log_p1, log_p2 = m.forward(wv_idx1, wv_idx2)

		# loss
		crit1 = torch.nn.NLLLoss(size_average=False)
		crit2 = torch.nn.NLLLoss(size_average=False)
		if opt.gpuid != -1:
			crit1 = crit1.cuda()
			crit2 = crit2.cuda()
		loss1 = crit1(log_p1, y_gold[:,0])	# loss on start idx
		loss2 = crit2(log_p2, y_gold[:,1])	# loss on end idx
		loss = (loss1 + loss2) / batch_l

		total_loss += loss.data[0] * batch_l

		# stats
		idx1 = pick_idx(log_p1.data)
		idx2 = pick_idx(log_p2.data)
		val_idx1_correct += count_correct_idx(idx1, y_gold[:,0].data)
		val_idx2_correct += count_correct_idx(idx2, y_gold[:,1].data)
		num_ex += batch_l

	m.end_pass()

	acc1 = float(val_idx1_correct) / num_ex
	acc2 = float(val_idx2_correct) / num_ex
	avg_loss = total_loss / num_ex
	return (acc1, acc2, avg_loss)



def main(args):
	opt = parser.parse_args(args)
	shared = Holder()

	torch.manual_seed(opt.seed)
	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
		torch.cuda.manual_seed_all(opt.seed)

	# build model
	m = Pipeline(opt, shared)
	optim = Optimizer(opt, shared)
	ema = EMA(opt, shared)

	m.init_weight()
	if opt.gpuid != -1:
		m = m.cuda()

	# loading data
	train_res_files = None if opt.train_res == '' else opt.train_res.split(',')
	val_res_file = None if opt.val_res == '' else opt.val_res.split(',')
	train_data = Data(opt, opt.train_data, train_res_files)
	val_data = Data(opt, opt.val_data, val_res_file)

	train(opt, shared, m, optim, ema, train_data, val_data)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))