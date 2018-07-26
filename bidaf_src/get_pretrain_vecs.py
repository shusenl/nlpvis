import numpy as np
import h5py
import re
import sys
import operator
import argparse

def load_glove_vec(fname, vocab):
    dim = 0
    word_vecs = {}
    for line in open(fname, 'r'):
        d = line.split()
        word = d[0]
        vec = np.array(map(float, d[1:]))
        dim = vec.size

        if word in vocab:
            word_vecs[word] = vec
    return word_vecs, dim

def main():
  parser = argparse.ArgumentParser(
      description =__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--dict', help="*.dict file", type=str,
                      default='data/squad.word.dict')
  parser.add_argument('--glove', help='pretrained word vectors', type=str, default='')
  parser.add_argument('--output', help="output hdf5 file", type=str,
                      default='data/glove.hdf5')
  
  args = parser.parse_args()
  vocab = open(args.dict, "r").read().split("\n")[:-1]
  vocab = map(lambda x: (x.split()[0], int(x.split()[1])), vocab)
  word2idx = {x[0]: x[1] for x in vocab}

  print("vocab size: " + str(len(vocab)))
  w2v, dim = load_glove_vec(args.glove, word2idx)
  print("matched word vector size: {0}, dim: {1}".format(len(w2v), dim))
  
  rs = np.random.normal(size = (len(vocab), dim))
      
  print("num words in pretrained model is " + str(len(w2v)))
  for word, vec in w2v.items():
      rs[word2idx[word]] = vec
  
  with h5py.File(args.output, "w") as f:
    f["word_vecs"] = np.array(rs)
    
if __name__ == '__main__':
    main()
