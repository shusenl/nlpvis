'''
store the NLP model inner states during the evaluation process

'''

#### for baseline performance
# from sklearn.neighbors import NearestNeighbors
# import bson
import numpy as np
import pickle

class hiddenStateRecorder:
    def __init__(self, filename=None):

        self.hiddenStore = {}
        self.dictEntry = {}
        #### for appendTagState
        self.currentTag = {}

        if filename:
            self.load(filename)

    def clear(self):
        self.hiddenStore = {}
        #### for appendTagState
        self.currentTag = {}
        self.dictEntry = {}

    '''
        record state corresponding to a string (word, sentence, label, tag)
        in the neural network for high-dimensional lookup
    '''
    def saveTagState(self, stateType, tag, states=None):
        if stateType not in self.hiddenStore:
            self.hiddenStore[stateType] = {}

        self.currentTag[stateType] = tag
        if states is not None:
            self.hiddenStore[stateType][tag] = states

    def saveDictEntry(self, key, value):
        self.dictEntry[key] = value

    '''
        record the state when the corresponding tag can not be accessed
        in the same context, i.e., when the original sentence can not be
        accessed in the encoder layer.
    '''
    def appendTagState(self, stateType, states):
        if stateType in self.currentTag:
            tag = self.currentTag[stateType]
            self.hiddenStore[stateType][tag] = states

    def save(self, outputPath):
        with open(outputPath, 'wb') as f:
            # f.write(bson.BSON.encode(self.hiddenStore))
            pickle.dump(self.hiddenStore, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, inputPath):
        with open(inputPath, 'rb') as f:
            # self.hiddenStore = bson.decode_all(f.read())
            self.hiddenStore = pickle.load(f)

    def buildSearchIndex(self, stateType=None):
        if stateType == None:
            stateType = self.hiddenStore.keys()[0]

        store = self.hiddenStore[stateType]
        self.hiddenStore[stateType] = {}
        size = len(store)
        print( "build index for ", size, " sentence ...")
        data = self.hiddenStore[stateType]["data"] = np.zeros( (store[store.keys()[0]].size, size) )
        sen2index = self.hiddenStore[stateType]["sen2index"] = {}
        index2sen = self.hiddenStore[stateType]["index2sen"] = []

        for index, key in enumerate(store.keys()):
            entry = store[key]
            # print entry
            data[:,index] = entry
            sen2index[key] = index
            index2sen.append(key)
        # print "data:", data.shape

    def neighborLookup(self, stateType, tag, k=20):
        # default baseline implementation
            neighbors = {}
        # try:
            sen2index = self.hiddenStore[stateType]["sen2index"]
            index2sen = self.hiddenStore[stateType]["index2sen"]
            # print "sen2index:", sen2index
            print ("index2sen:", index2sen)
            print ("keys:", sen2index.keys())

            index = sen2index[tag]
            data = self.hiddenStore[stateType]["data"]
            ref = data[:,index]
            # print np.squeeze(np.array(np.matrix(data).T*np.matrix(ref).T))
            #euclidean distance
            dist = np.linalg.norm(data-np.vstack(ref), axis=0)
            indices = np.argsort(dist)[0:k]
            #cosine distance
            # indices = np.argsort(np.squeeze(np.array(np.matrix(data).T*np.matrix(ref).T)))
            indices = indices[0:k]
            # print indices
            neighbors["sentence"] = [index2sen[ind] for ind in indices]
            neighbors["distance"] = [dist[ind] for ind in indices]
            neighbors["prediction"] = [ self.dictEntry[index2sen[ind]] for ind in indices]
            return neighbors
        # except:
            # print "No index structure is built"

        # print indices[0:20]
        # return neighbors
