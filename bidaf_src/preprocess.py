import os
import sys
import argparse
import numpy as np
import h5py
import itertools
from collections import defaultdict
import json


class Indexer:
    def __init__(self, symbols = ["<pad>"]):
        self.PAD = symbols[0]
        self.num_oov = 1
        self.d = {self.PAD: 0}
        for i in range(self.num_oov): #hash oov words to one of 100 random embeddings
            oov_word = '<oov'+ str(i) + '>'
            self.d[oov_word] = len(self.d)
            
    def convert(self, w):        
        return self.d[w] if w in self.d else self.d['<oov' + str(np.random.randint(self.num_oov)) + '>']

    def convert_sequence(self, ls):
        return [self.convert(l) for l in ls]

    def write(self, outfile):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.d.iteritems()]
        items.sort()
        for v, k in items:
            print >>out, k, v                
        out.close()

    def register_words(self, wv, seq):
        for w in seq:
            if w in wv and w not in self.d:
                self.d[w] = len(self.d)


class CharIndexer:
    def __init__(self, symbols = ["<blank>"]):
        self.PAD = symbols[0]
        self.d = {self.PAD: 0}

    def convert(self, ls):
        return [self.d[w] for l in ls]

    def write(self, outfile):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.d.iteritems()]
        items.sort()
        for v, k in items:
            print >>out, k, v                
        out.close()

    def register_char(self, word):
        for c in word:
            if c not in self.d:
                self.d[c] = len(self.d)


def pad(ls, length, symbol, pad_back = True):
    if len(ls) >= length:
        return ls[:length]
    if pad_back:
        return ls + [symbol] * (length -len(ls))
    else:
        return [symbol] * (length -len(ls)) + ls  


def get_glove_words(f):
    glove_words = set()
    for line in open(f, "r"):
        word = line.split()[0].strip()
        glove_words.add(word)
    return glove_words


def make_vocab(args, glove_vocab, word_indexer, srcfile, targetfile, seqlength):
    num_ex = 0
    found_toks = set()
    all_toks = set()
    for _, (src_orig, targ_orig) in enumerate(itertools.izip(open(srcfile,'r'), open(targetfile,'r'))):
        if args.lowercase == 1:
            src_orig = src_orig.lower()
            targ_orig = targ_orig.lower()

        targ = targ_orig.strip().split()
        src = src_orig.strip().split()

        assert(len(targ) <= seqlength and len(src) <= seqlength)

        num_ex += 1
        word_indexer.register_words(glove_vocab, targ)
        for word in targ:
            all_toks.add(word)
            
            if word in word_indexer.d:
                found_toks.add(word)

        word_indexer.register_words(glove_vocab, src)
        for word in src:
            all_toks.add(word)
            if word in word_indexer.d:
                found_toks.add(word)
            
    return num_ex, len(found_toks), len(all_toks)


def make_char_vocab(args, char_indexer, srcfile, targetfile, wordlength):
    num_ex = 0
    for _, (src_orig, targ_orig) in enumerate(itertools.izip(open(srcfile,'r'), open(targetfile,'r'))):
        if args.lowercase == 1:
            src_orig = src_orig.lower()
            targ_orig = targ_orig.lower()

        targ = targ_orig.strip().split()
        src = src_orig.strip().split()

        for word in targ:
            for c in word:
                char_indexer.add(c)

        for word in src:
            for c in word:
                char_indexer.add(c)

    return num_ex



def convert(args, word_indexer, srcfile, targetfile, spanfile, batchsize, seqlength, outfile, num_ex, min_sent_l=10000, max_sent_l=0, seed=0):
    np.random.seed(seed)

    targets = np.zeros((num_ex, seqlength), dtype=int)
    sources = np.zeros((num_ex, seqlength), dtype=int)
    source_lengths = np.zeros((num_ex,), dtype=int)
    target_lengths = np.zeros((num_ex,), dtype=int) # target sentence length (1 sentence)
    spans = np.zeros((num_ex, 2), dtype=int)
    batch_keys = np.array([None for _ in range(num_ex)])
    ex_idx = np.zeros(num_ex, dtype=int)

    dropped = 0
    sent_id = 0
    for _, (src_orig, targ_orig, span_orig) in enumerate(itertools.izip(open(srcfile,'r'), open(targetfile,'r'), open(spanfile,'r'))):
        if args.lowercase == 1:
            src_orig = src_orig.lower()
            targ_orig = targ_orig.lower()

        src = src_orig.strip().split()
        targ =  targ_orig.strip().split()
        span = span_orig.strip().split()
        assert(len(span) == 2)
        span = [int(span[0]), int(span[1])] # end idx is inclusive
        
        min_sent_l = min(len(targ), len(src), min_sent_l)
        max_sent_l = max(len(targ), len(src), max_sent_l)
        # DO NOT drop anything, causing inconsistent indices

        # pad to meet seqlength
        targ = pad(targ, seqlength, word_indexer.PAD)
        targ = word_indexer.convert_sequence(targ)
        targ = np.array(targ, dtype=int)
        src = pad(src, seqlength, word_indexer.PAD)
        src = word_indexer.convert_sequence(src)
        src = np.array(src, dtype=int)
        span = np.array(span, dtype=int)
        
        targets[sent_id] = np.array(targ,dtype=int)
        target_lengths[sent_id] = (targets[sent_id] != 0).sum()
        sources[sent_id] = np.array(src, dtype=int)
        source_lengths[sent_id] = (sources[sent_id] != 0).sum() 
        spans[sent_id] = np.array(span, dtype=int)
        batch_keys[sent_id] = (source_lengths[sent_id], target_lengths[sent_id])

        # sanity check
        assert(spans[sent_id][0] < source_lengths[sent_id] and spans[sent_id][1] < source_lengths[sent_id])

        sent_id += 1
        if sent_id % 10000 == 0:
            print("{}/{} sentences processed".format(sent_id, num_ex))

    assert(sent_id == num_ex)
    print("{}/{} sentences processed".format(sent_id, num_ex))
    # shuffle
    rand_idx = np.random.permutation(num_ex)
    targets = targets[rand_idx]
    sources = sources[rand_idx]
    spans = spans[rand_idx]
    source_lengths = source_lengths[rand_idx]
    target_lengths = target_lengths[rand_idx]
    batch_keys = batch_keys[rand_idx]
    ex_idx = rand_idx
    
    # break up batches based on source/target lengths
    sorted_keys = sorted([(i, p) for i, p in enumerate(batch_keys)], key=lambda x: x[1])
    sorted_idx = [i for i, _ in sorted_keys]
    # rearrange examples
    sources = sources[sorted_idx]
    targets = targets[sorted_idx]
    spans = spans[sorted_idx]
    target_l = target_lengths[sorted_idx]
    source_l = source_lengths[sorted_idx]
    ex_idx = rand_idx[sorted_idx]

    cur_src_l = -1
    cur_tgt_l = -1
    batch_location = [] #idx where src/targ length changes
    for j,i in enumerate(sorted_idx):
        if batch_keys[i][0] != cur_src_l or batch_keys[i][1] != cur_tgt_l:
            cur_src_l = batch_keys[i][0]
            cur_tgt_l = batch_keys[i][1]
            batch_location.append(j)

    # get batch strides
    cur_idx = 0
    batch_idx = [0]
    batch_l = []
    target_l_new = []
    source_l_new = []
    for i in range(len(batch_location)-1):
        end_location = batch_location[i+1]
        while cur_idx < end_location:
            cur_idx = min(cur_idx + batchsize, end_location)
            batch_idx.append(cur_idx)

    # rearrange examples according to batch strides
    for i in range(len(batch_idx)):
        end = batch_idx[i+1] if i < len(batch_idx)-1 else len(sources)

        batch_l.append(end - batch_idx[i])
        source_l_new.append(source_l[batch_idx[i]])
        target_l_new.append(target_l[batch_idx[i]])

        # sanity check
        for k in range(batch_idx[i], end):
            assert(source_l[k] == source_l_new[-1])
            assert(target_l[k] == target_l_new[-1])
            assert(sources[k, source_l[k]:].sum() == 0)
            assert(targets[k, target_l[k]:].sum() == 0)


    # Write output
    f = h5py.File(outfile, "w")        
    f["source"] = sources
    f["target"] = targets
    f["target_l"] = target_l_new    # (#batch,)
    f["source_l"] = source_l_new    # (#batch,)
    f["span"] = spans
    f["batch_l"] = batch_l
    f["batch_idx"] = batch_idx
    f["source_size"] = np.array([len(word_indexer.d)])
    f["target_size"] = np.array([len(word_indexer.d)])
    f['ex_idx'] = ex_idx
    print("Saved {} sentences (dropped {} due to length/unk filter)".format(
        len(f["source"]), dropped))
    print('Number of batches: {0}'.format(len(batch_idx)))
    f.close()                
    return min_sent_l, max_sent_l

def process(args):
    word_indexer = Indexer()
    glove_vocab = get_glove_words(args.glove)

    print("First pass through data to get vocab...")
    num_ex_train, found_tok_cnt, all_tok_cnt = make_vocab(args, glove_vocab, word_indexer, args.srcfile, args.targfile, args.seqlength)
    print("Number of sentences in training: {0}, number of tokens: {1}/{2}".format(num_ex_train, found_tok_cnt, all_tok_cnt))
    num_ex_valid, found_tok_cnt, all_tok_cnt = make_vocab(args, glove_vocab, word_indexer, args.srcvalfile, args.targvalfile, args.seqlength)
    print("Number of sentences in valid: {0}, number of tokens: {1}/{2}".format(num_ex_valid, found_tok_cnt, all_tok_cnt))   
    
    print('Number of tokens collected: {0}'.format(len(word_indexer.d)))
    word_indexer.write(args.outputfile + ".word.dict")

    min_sent_l = 1000000
    max_sent_l = 0
    min_sent_l, max_sent_l = convert(args, word_indexer, args.srcvalfile, args.targvalfile, args.spanvalfile, args.batchsize, args.seqlength, args.outputfile + "-val.hdf5", num_ex_valid,
                         min_sent_l, max_sent_l, args.seed)
    min_sent_l, max_sent_l = convert(args, word_indexer, args.srcfile, args.targfile, args.spanfile, args.batchsize, args.seqlength, args.outputfile + "-train.hdf5", num_ex_train, min_sent_l, max_sent_l, args.seed)
    print("Min sent length (before dropping): {}".format(min_sent_l))
    print("Max sent length (before dropping): {}".format(max_sent_l))    
    
def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vocabsize', help="Size of source vocabulary, constructed "
                                                "by taking the top X most frequent words.",
                                                type=int, default=50000)
    parser.add_argument('--srcfile', help="Path to sent1 training data.",
                        default = "data/train.context.txt")
    parser.add_argument('--targfile', help="Path to sent2 training data.",
                        default = "data/train.query.txt")
    parser.add_argument('--spanfile', help="Path to span data.",
                        default = "data/train.span.txt")    
    parser.add_argument('--srcvalfile', help="Path to sent1 validation data.",
                        default = "data/dev.context.txt")
    parser.add_argument('--targvalfile', help="Path to sent2 validation data.",
                        default = "data/dev.query.txt")
    parser.add_argument('--spanvalfile', help="Path to span validation data.",
                        default = "data/dev.span.txt")
    
    parser.add_argument('--batchsize', help="Size of each minibatch.", type=int, default=16)
    parser.add_argument('--seqlength', help="Maximum sequence length. Sequences longer than this are dropped.", type=int, default=800)
    parser.add_argument('--wordlength', help="Maximum sequence length. Sequences longer than this are dropped.", type=int, default=50)
    parser.add_argument('--outputfile', help="Prefix of the output file names. ",
                        type=str, default = "data/squad")
    parser.add_argument('--lowercase', help="Whether to use lowercase for vocabulary", type=int, default = 1)
    parser.add_argument('--seed', help="seed of shuffling sentences before sorting (based on  "
                                           "source length).", type = int, default = 1)
    parser.add_argument('--glove', type = str, default = '')    
    args = parser.parse_args(arguments)
    process(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
