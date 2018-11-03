#import ujson
#
#path = './data/dev-v1.1.json'
#
#with open(path, 'r') as f:
#	f_str = f.read()
#j_obj = ujson.loads(f_str)
#
#
#
#print(len(j_obj['data']))
#for k,v in j_obj['data'][0].iteritems():
#	print(k)
#
#print(len(j_obj['data'][0]['paragraphs']))
#for i in range(len(j_obj['data'][0]['paragraphs'])):
#	par_i = j_obj['data'][0]['paragraphs'][i]
#	print('context', par_i['context'])
#	context = par_i['context']
#
#	qas = par_i['qas']
#	for qa in qas:
#		print('question', qa['question'])
#		
#		ans = qa['answers']
#		for a in ans:
#			print('answer_start', a['answer_start'])
#			print('text', a['text'])
#
#			idx1 = a['answer_start']
#			idx2 = len(a['text']) + idx1
#			print(context[idx1:idx2])
#
#
#	assert(False)
#	
#
#

#
#import nltk
#def tokenize(seq):
#	toks = nltk.word_tokenize(seq)
#	return toks
#
#
#def map_answer_idx(context_toks, answer_toks, char_idx1):
#	context = ' '.join(context_toks)
#	answer = ' '.join(answer_toks)
#	new_char_idx1 = context[char_idx1:].index(answer) + char_idx1
#	new_char_idx2 = new_char_idx1 + len(answer)
#
#	# count number of spaces
#	tok_idx1 = context[new_char_idx1::-1].count(' ')
#	tok_idx2 = context[new_char_idx2::-1].count(' ')
#	return (tok_idx1, tok_idx2)
#
#def remap_char_idx(context, context_toks, char_idx):
#	context_tok_seq = ' '.join(context_toks)
#	m = [-1 for _ in range(len(context))]
#	i = 0
#	j = 0
#	while (i < len(context) and j < len(context_tok_seq)):
#		# skip white spaces
#		while context[i] == ' ':
#			i += 1
#		while context_tok_seq[j] == ' ':
#			j += 1
#
#		if context[i] == context_tok_seq[j]:
#			m[i] = j
#			i += 1
#			j += 1
#		else:
#			assert(False)
#
#	return m
#
#
#context = 'Mary\'s apple falls into John\'s hand'
#answer = 'John\'s hand'
#char_idx1 = context.index(answer)
#
#context_toks = tokenize(context)
#answer_toks = tokenize(answer)
#
#context_tok_seq = ' '.join(context_toks)
#answer_tok_seq = ' '.join(answer_toks)
#
#print('context tokenized')
#print(context_tok_seq)
#print('answer tokenized')
#print(answer_tok_seq)
#print('char idx1', char_idx1)
#
#(idx1, idx2) = map_answer_idx(context_toks, answer_toks, char_idx1)
#print('target span', idx1, idx2)
#
#
#m = remap_char_idx(context, context_toks, idx1)
#print(m)
#print(context_tok_seq[m[char_idx1]:])
#
#

#import torch
#from torch import nn
#from torch.autograd import Variable

#def get_span_f1(pred, gold):
#	pred = pred.cpu()
#	gold = gold.cpu()
#	gold_start = gold[:,0]
#	gold_end = gold[:,1] + 1	# exclusive ending
#	pred_start = pred[:,0]
#	pred_end = pred[:,1] + 1	# exclusive ending
#
#	start = torch.max(pred_start, gold_start)
#	end = torch.min(pred_end, gold_end)
#
#	pred_range = pred_end - pred_start
#	gold_range = gold_end - gold_start
#	overlap = (end - start).clamp(min=0)
#
#	# recall
#	rec = overlap.float() / gold_range.float()
#
#	# numerical fixes for precision
#	pred_range = pred_range.clamp(min=0)
#	denom = pred_range.float()
#	denom_mask = (pred_range == 0).float()
#	prec = overlap.float() / (denom + denom_mask)
#	
#	# numerical fixes for f1
#	denom = prec + rec
#	denom_mask = (denom == 0.0).float()
#	f1 = 2.0 * prec * rec / (denom + denom_mask)
#
#	return (prec, rec, f1)
#
#
#pred = torch.Tensor([1,2,0,4,3,5,6,5]).view(4,2)
#gold = torch.Tensor([1,1,1,4,3,5,1,2]).view(4,2)
#
#acc = get_span_f1(pred, gold)
#print(acc)


#import torch
#from torch import nn
#from torch.autograd import Variable
#
#batch_l = 2
#seq_l = 7
#W = 7
#in_channel = 4
#out_channel = 6
#height = 1
#width = 5
#conv = nn.Conv2d(in_channel, out_channel, (height, width))
#
#b = Variable(torch.randn(batch_l, in_channel, seq_l, W))
#print(b)
#c = conv(b)
#print(c)

import torch
from torch import nn
from torch.autograd import Variable

embedding_dim = 16
num_filters = 100
ngram_size = 5
conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=ngram_size)

batch_l = 2
seq_l = 10
x = Variable(torch.randn(batch_l, embedding_dim, seq_l))
print(x)
y = conv(x)
print(y)


#import numpy as np
#import re
#import sys
#
#def load_glove_vec(fname):
#    dim = 0
#    word_vecs = {}
#    with open(fname, 'r+') as f:
#    	for line in f:
#    	    d = line.split()
#    	    word = d[0]
#    	    vec = np.array(map(float, d[1:]))
#    	    dim = vec.size
#	
#    	    word_vecs[word] = vec
#    return word_vecs, dim

