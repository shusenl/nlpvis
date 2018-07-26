'''
Test the model
python2.7 eval.py --gpuid -1 --data ../data/bidaf/squad-val.hdf5 --word_vecs ../data/bidaf/glove.hdf5 --rnn_type lstm --word_vec_size 300 --load_file ../data/bidaf/bidaf_5.ema
'''

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
from embeddings import *
import nltk
from data import *

class bidafModelInterface:
    def __init__(self, wordDict, wordVec, model):
        opt = argparse.Namespace()
        opt.gpuid = -1
        opt.word_vec_size = 300
        opt.word_vecs = wordVec
        opt.load_file = model
        opt.dict = wordDict

        opt.hidden_size = 200
        opt.dropout = 0.2
        opt.dropout_h = 0.2

        opt.fix_word_vecs = 1
        opt.enc_rnn_layer =1
        opt.reenc_rnn_layer =2
        opt.cls_rnn_layer =1
        opt.birnn = 1
        opt.rnn_type = 'lstm'
        opt.hw_layer = 2

        self.shared = Holder()

        self.tokenMap = {}
        with open(wordDict, 'r+') as f:
            lines = f.readlines()
            for l in lines:
                toks = l.split(" ")
                self.tokenMap[toks[0]] = int(toks[1])
        # print self.tokenMap

        if opt.gpuid != -1:
            torch.cuda.set_device(opt.gpuid)

        # build model
        self.m = Pipeline(opt, self.shared)
        self.m.train(False)

        # initialization
        self.opt = opt
        self.reloadModel()

        if opt.gpuid != -1:
            self.m = self.m.cuda()

    def reloadModel(self):
        # initialization
        print('loading pretrained model from {0}...'.format(self.opt.load_file))
        param_dict = load_param_dict('{0}.hdf5'.format(self.opt.load_file))
        self.m.set_param_dict(param_dict)

        # model_parameters = filter(lambda p: p.requires_grad, self.m.parameters())
        # num_params = sum([np.prod(p.size()) for p in model_parameters])
        # print('total number of trainable parameters: {0}'.format(num_params))

    def mapToToken(self, sentence):
        tokenList = []
        sentence = sentence.rstrip().split(" ")
        for word in sentence:
            if word in self.tokenMap.keys():
                tokenList.append(self.tokenMap[word])
            else:
                tokenList.append(self.tokenMap["<oov0>"])
        # print tokenList
        token = torch.LongTensor(tokenList).view(1, len(tokenList))
        return token

    def attention(self, att_name='att_soft1'):
        # print dir(self.shared)
        # print "att_name:", att_name
        batch_att = getattr(self.shared, att_name)
        print self.shared.keys()
        att = batch_att.data[0, 0:, 0:]
        att = att.numpy()
        # print "attention range:", att.min(), att.max()
        # att = att/att.max()
        return att

    def predict(self, sentencePair):
        #map to token
        sourceSen = sentencePair[0]
        targetSen = sentencePair[1]
        if sourceSen and targetSen:
            source = self.mapToToken(sourceSen)
            target = self.mapToToken(targetSen)
            print source, "\n"
            print target, "\n"

            wv_idx1 = Variable(source, requires_grad=False)
            wv_idx2 = Variable(target, requires_grad=False)
            # set resources, TODO

            # update network parameters
            self.m.update_context([0], 1, source.shape[1], target.shape[1])
            # self.m.update_context([0], 1, source.shape[1], target.shape[1], res_map, raw)

            # forward pass
            log_p1, log_p2 = self.m.forward(wv_idx1, wv_idx2)
            # print log_p1, log_p2
            p1 = log_p1.exp().data.numpy()
            p2 = log_p2.exp().data.numpy()
            startIndex = np.argmax(p1)
            endIndex = np.argmax(p2)
            print "startIndex, endIndex", startIndex, endIndex
            words = sourceSen.rstrip().split(" ")
            prediction = ",".join(words[startIndex:endIndex+1])
            print "prediction:", prediction
            return [p1.tolist(), p2.tolist()]

            # loss
            # crit1 = torch.nn.NLLLoss(size_average=False)
            # crit2 = torch.nn.NLLLoss(size_average=False)
            # if opt.gpuid != -1:
            #     crit1 = crit1.cuda()
            #     crit2 = crit2.cuda()
        # loss1 = crit1(log_p1, y_gold[:,0])    # loss on start idx
        # loss2 = crit2(log_p2, y_gold[:,1])    # loss on end idx
        # loss = (loss1 + loss2) / batch_l
