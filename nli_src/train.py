import sys
sys.path.insert(0, '../')
from pipeline import *
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
from holder import *
from optimizer import *
from embeddings import *
from data import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--train_data', help="Path to training data hdf5 file.", default="data/entail-train.hdf5")
parser.add_argument('--val_data', help="Path to validation data hdf5 file.", default="data/entail-val.hdf5")
parser.add_argument('--test_data', help="Path to test data hdf5 file.", default="data/entail-test.hdf5")

parser.add_argument('--save_file', help="Path to where model to be saved.", default="model")

## pipeline specs
parser.add_argument('--encoder', help="The type of encoder", default="proj")
parser.add_argument('--attention', help="The type of attention", default="local")
parser.add_argument('--classifier', help="The type of classifier", default="local")
parser.add_argument('--hidden_size', help="The hidden size of the pipeline", type=int, default=200)
parser.add_argument('--word_vec_size', help="The input word embedding dim", type=int, default=300)
parser.add_argument('--dropout', help="The dropout probability", type=float, default=0.2)
parser.add_argument('--num_att_labels', help='The number of attention labels', type=int, default=1)
parser.add_argument('--num_labels', help="The number of prediction labels", type=int, default=3)
parser.add_argument('--constr', help="The list of constraints to use in hard attention", default='')
#parser.add_argument('--percent', help="The percent of training data to use", type=float, default=1.0)
parser.add_argument('--epochs', help="The number of epoches for training", type=int, default=100)
# TODO, param_init of uniform dist or normal dist???
parser.add_argument('--param_init_type', help="The type of parameter initialization", default='uniform')
parser.add_argument('--param_init', help="The scale of the normal distribution from which weights are initialized", type=float, default=0.01)
parser.add_argument('--learning_rate', help="The learning rate for training", type=float, default=0.05)
parser.add_argument('--word_vecs', help="The path to word embeddings", default = "")
parser.add_argument('--fix_word_vecs', help="Whether to make word embeddings NOT learnable", type=int, default=1)
parser.add_argument('--print_every', help="Print stats after this many batches", type=int, default=1000)
parser.add_argument('--seed', help="The random seed", type=int, default=3435)
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--customize_att', help="Whether to use customized att values instead of computed ones", type=int, default=0)

def pick_label(dist):
	return np.argmax(dist, axis=1)

def train_batch(opt, shared, wv, m, optim, data, epoch_id):
	train_loss = 0.0
	num_ex = 0
	batch_order = torch.randperm(data.size())
	start_time = time.time()
	num_words_source = 0
	num_words_target = 0
	train_num_correct = 0

	# loss function
	criterion = torch.nn.NLLLoss(size_average=False)
	if opt.gpuid != -1:
		criterion = criterion.cuda()

	m.train(True)
	for i in xrange(data.size()):
		data_name, source, target, batch_ex_idx, batch_l, source_l, target_l, label = data[batch_order[i]]

		wv_idx1 = Variable(source, requires_grad=False)
		wv_idx2 = Variable(target, requires_grad=False)
		y_gold = Variable(label, requires_grad=False)
		# set resources, TODO

		# lookup word vecs
		word_vecs1 = wv(wv_idx1)
		word_vecs2 = wv(wv_idx2)

		# update network parameters
		m.update_context(batch_ex_idx, batch_l, source_l, target_l)

		# forward pass
		#t1 = time.time()
		y_dist = m.forward(word_vecs1, word_vecs2)
		loss = criterion(y_dist, y_gold)
		#t2 = time.time()
		#print('{0}s used in forward'.format(t2-t1))

		# stats
		train_num_correct += np.equal(pick_label(y_dist.data), label).sum()

		# backward pass
		m.zero_grad()
		#t1 = time.time()
		loss.backward()
		#t2 = time.time()
		#print('{0}s used in backward'.format(t2-t1))

		# update network weights
		# TODO, for fix_word_vecs=0, need another optim for embeddings
		optim.step(shared)

		# printing
		num_words_source += batch_l * source_l
		num_words_target += batch_l * target_l
		train_loss += loss.data[0]
		num_ex += batch_l
		time_taken = time.time() - start_time
		if (i+1) % opt.print_every == 0:
			stats = 'Epoch: {0}, Batch: {1}/{2}, Batch size: {3}, LR: {4:.4f}, '.format(epoch_id+1, i+1, data.size(), batch_l, opt.learning_rate)
			stats += 'Loss: {0:.4f}, Acc: {1:.4f}, '.format(train_loss / num_ex, float(train_num_correct) / num_ex)
			stats += 'Speed: {0:.4}k, Time: {1:.4f}'.format(float(num_words_source + num_words_target) / time_taken / 1000, time_taken)
			print(stats)

	return train_loss, num_ex, train_num_correct

def train(opt, shared, wv, m, optim, train_data, valid_data):
	best_val_perf = 0.0
	test_perf = 0.0
	train_perfs = []
	val_perfs = []

	for i in xrange(opt.epochs):
		loss, num_ex, num_correct = train_batch(opt, shared, wv, m, optim, train_data, i)
		train_acc = float(num_correct) / num_ex
		train_perfs.append(train_acc)
		print('Train {0:.4f}'.format(train_acc))

		# evaluate
		#	and save if it's the best model
		val_acc = validate(opt, shared, wv, m, valid_data)
		val_perfs.append(val_acc)
		print('Val {0:.4f}'.format(val_acc))

		str_perf_table = ''
		for i in xrange(len(train_perfs)):
			str_perf_table += '{0}\t{1:.6f}\t{2:.6f}\n'.format(i+1, train_perfs[i], val_perfs[i])
		print(str_perf_table)

		if val_acc > best_val_perf:
			best_val_perf = val_acc
			print('saving model to {0}'.format(opt.save_file))
			param_dict = m.get_param_dict()
			save_param_dict(param_dict, '{0}.hdf5'.format(opt.save_file))
		else:
			print('skip saving model for perf <= {0:.4f}'.format(best_val_perf))



def validate(opt, shared, wv, m, data):
	m.train(False)

	total_loss = 0.0
	num_sents = 0
	num_correct = 0
	# loss function
	criterion = torch.nn.NLLLoss(size_average=False)
	if opt.gpuid != -1:
		criterion = criterion.cuda()

	for i in xrange(data.size()):
		data_name, source, target, batch_ex_idx, batch_l, source_l, target_l, label = data[i]

		wv_idx1 = Variable(source, requires_grad=False)
		wv_idx2 = Variable(target, requires_grad=False)
		y_gold = Variable(label, requires_grad=False)
		# set resources, TODO

		# lookup word vecs
		word_vecs1 = wv(wv_idx1)
		word_vecs2 = wv(wv_idx2)

		# update network parameters
		m.update_context(batch_ex_idx, batch_l, source_l, target_l)

		# forward pass
		y_dist = m.forward(word_vecs1, word_vecs2)
		loss = criterion(y_dist, y_gold)
		total_loss += loss.data[0]

		# stats
		num_correct += np.equal(pick_label(y_dist.data), label).sum()
		num_sents += batch_l

	acc = float(num_correct) / num_sents
	return acc



def main(args):
	opt = parser.parse_args(args)
	shared = Holder()

	torch.manual_seed(opt.seed)
	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
		torch.cuda.manual_seed_all(opt.seed)

	# build model
	embeddings = WordVecLookup(opt)
	pipeline = Pipeline(opt, shared)
	optim = Adagrad(pipeline, opt)

	# initialize model
	pipeline.init_weight()
	if opt.gpuid != -1:
		embeddings.cuda()
		pipeline = pipeline.cuda()
		optim.cuda()

	# loading data
	train_data = Data(opt, opt.train_data)
	val_data = Data(opt, opt.val_data)
	train(opt, shared, embeddings, pipeline, optim, train_data, val_data)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
