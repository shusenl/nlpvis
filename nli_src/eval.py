import sys
sys.path.insert(0, '../')
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
from .pipeline import *
from .holder import *
from .optimizer import *
from .embeddings import *
from .data import *
from .util import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data', help="Path to evaluation data hdf5 file.", default="data/entail-val.hdf5")
parser.add_argument('--res', help="Path to resource files, separated by comma.", default="")
parser.add_argument('--load_file', help="Path from where model to be loaded.", default="model")

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
parser.add_argument('--word_vecs', help="The path to word embeddings", default = "")
parser.add_argument('--fix_word_vecs', help="Whether to make word embeddings NOT learnable", type=int, default=1)
parser.add_argument('--seed', help="The random seed", type=int, default=3435)
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--customize_att', help="Whether to use customized att values instead of computed ones", type=int, default=0)

def pick_label(dist):
	return np.argmax(dist, axis=1)

def evaluate(opt, shared, wv, m, data):
	m.train(False)

	total_loss = 0.0
	num_sents = 0
	num_correct = 0
	# loss function
	criterion = torch.nn.NLLLoss(size_average=False)
	if opt.gpuid != -1:
		criterion = criterion.cuda()

	predictionValue = []
	groundTruthLabel = []
	# number of batches
	for i in range(data.size()):
		data_name, source, target, batch_ex_idx, batch_l, source_l, target_l, label = data[i]
		# print "source, target: ", source, target

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
		# print y_dist.exp(), y_gold
		pred = y_dist.exp().data.cpu().numpy() if opt.gpuid != -1 else y_dist.exp().data.numpy()
		# print pred.tolist(), pred.shape
		if pred.shape[0] == 1:
			predictionValue.append(pred.tolist())
		else:
			predictionValue += pred.tolist()

		predLabel = label.cpu().numpy() if opt.gpuid != -1 else label.numpy()
		groundTruthLabel += predLabel.tolist()

		loss = criterion(y_dist, y_gold)
		total_loss += loss.data[0]

		# stats
		num_correct += np.equal(pick_label(y_dist.data), label).sum()
		num_sents += batch_l

	acc = float(num_correct) / num_sents
	avg_loss = total_loss / num_sents
	print( "total sentence count:", num_sents)
	print( "total prediction count:", len(predictionValue))
	print( "total groundTruthLabel count:", len(groundTruthLabel))

	return (acc, avg_loss, predictionValue, groundTruthLabel)


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

	# initialize
	print('initializing model from {0}'.format(opt.load_file))
	param_dict = load_param_dict('{0}.hdf5'.format(opt.load_file))
	pipeline.set_param_dict(param_dict)
	if opt.gpuid != -1:
		embeddings.cuda()
		pipeline = pipeline.cuda()

	# loading data
	res_files = None if opt.res == '' else opt.res.split(',')
	data = Data(opt, opt.data)
	# data = Data(opt, opt.data, res_files)
	acc, avg_loss, _, _ = evaluate(opt, shared, embeddings, pipeline, data)
	print('Val: {0:.4f}, Loss: {1:.4f}'.format(acc, avg_loss))


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
