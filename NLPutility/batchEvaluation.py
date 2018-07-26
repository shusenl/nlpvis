'''
    batch test and summarize the data prediction resutl

    require:
        - input pairs
        - model hooks
        - perturbation hooks
'''
import sys
sys.path.insert(0, '..')

from nli_src.modelInterface import *
from sentenceGenerator import *
from hiddenStateRecorder import *
import pickle
import itertools

labels = ["entailment", "neutral", "contradiction"]

############################## batch evaluation ################################
class batchEvaluation:
    def __init__(self, srcFile, targFile, labelFile, saveFileName=None):
        #load input pair and grouth truth label
        self.srcFile = srcFile
        self.targFile = targFile
        self.labelFile = labelFile
        ### store the computed hidden states
        self.hiddenStore = hiddenStateRecorder()

        if saveFileName:
            self.saveFileName = saveFileName

    def initialize(self):
        if self.saveFileName:
            try:
                with open(self.saveFileName, 'rb') as handle:
                    self.storage = pickle.load(handle)
                    print self.storage.keys()
            except:
                print "File:", self.saveFileName, "does not exist. Generate prediction now ..."
                self.generatePerturbedPrediction()
        #     self.storage = saveFile
        # perturbation type: only perturb target/ perturb all


    def setSentencePerturbationHook(self, perturb):
        self.perturb = perturb

    def setPredictionHook(self, predict):
        self.predict =  predict

    def setAttentionHook(self, att):
        self.att = att

    '''
        this should also provide batch attentions
    '''
    def setBatchPredictionHook(self, batchPredict):
        self.batchPredict = batchPredict

    def setSentenceVerifyHook(self, verify):
        self.verify = verify

    def saveHiddenEncoding(self, outputPath):
        # self.hiddenStore.save(outputPath)
        # NLPutility.__module__ = "NLPutility"
        self.hiddenStore.save(outputPath)
        # with open(outputPath, 'wb') as handle:
            # pickle.dump(self.hiddenStore, handle, protocol=pickle.HIGHEST_PROTOCOL)

    '''
        generate statistics and write to a JSON file
        store as hierarchy [origin]->[perturb pairs]

        Label Filter
            - Original Prediction T/F
            - Prediction Change E/C/N=>E/C/N (for failed origin pair)

        Value Filter
            - Perturbation Change Percentage
            - Prediction deviation (KL divergence) mean
            - Prediction deviation (KL divergence) variance

    '''
    def generateStatistic(self, outputPath):
        ## per original pair ##
        # self.storage["perturbErrorRatio"] = []
        ## self.storage["originPredCase"] = []
        ## per all(perturbed+original) pairs ##
        # self.storage["predCase"] = []
        preOrigIndex = None
        origIndex = None

        count = 0
        wrongPred = 0
        allPred = 0

        # json output only have per original pair information
        self.jsonOut = []

        trueLabel = None
        currentPredLabel = None
        predLabel = None
        ratio = 0
        for index, origIndex in enumerate(self.storage["mapToOrigIndex"]):

            if preOrigIndex != None and preOrigIndex != origIndex:
                count = count + 1
                if allPred != 0:
                    ratio = 1.0-float(wrongPred)/float(allPred)
                else:
                    ratio = 0.0

                predLabel = labels[np.argmax(self.storage["origPred"][preOrigIndex])]
                #record previous entry
                item = {
                    "index": preOrigIndex,
                    "src": self.storage["origSrc"][preOrigIndex],
                    "targ": self.storage["origTarg"][preOrigIndex],
                    "stability": ratio,
                    "predict": trueLabel+'-'+predLabel,
                    "correctness": trueLabel == predLabel,
                    "trueLabel":trueLabel,
                    "perturbCount":allPred
                }
                self.jsonOut.append(item)

                #### reset variable ####
                wrongPred = 0
                allPred = 0
            else:
                currentPredLabel = labels[np.argmax(self.storage["pred"][index])]
                trueLabel = self.storage["origLabel"][origIndex]

                if currentPredLabel != trueLabel:
                    wrongPred = wrongPred + 1

                allPred = allPred + 1

            preOrigIndex = origIndex

            # print "original pair count: ", count
            # if count > 1:
            #     break
        ####### output json ##########
        import json
        with open(outputPath, 'w') as outfile:
                json.dump(self.jsonOut, outfile)


    def generateHiddenStates(self):
        num_lines = sum(1 for line in open(self.labelFile))
        index = 0
        for _, (src_orig, targ_orig, label_orig) in \
            enumerate(itertools.izip(open(self.srcFile,'r'),
            open(self.targFile,'r'),open(self.labelFile,'r'))):
                # generate perturbation
                # print index, src_orig, targ_orig, label_orig
                label_orig = label_orig.rstrip('\n')
                targ_perb = self.perturb(targ_orig)
                src_perb = self.perturb(src_orig)

                # if self.verify(src_orig) and self.verify(targ_orig):
                prediction = self.predict([src_orig,targ_orig], self.hiddenStore)
                # if index > 5:
                    # break

                index = index + 1
                ####### test on small number of example #####
                # if index > 100:
                    # break

                # batch prediction
                if index % 20 == 0:
                    print "  processing:", str(float(index)/float(num_lines)*100.0), str(index)

        self.hiddenStore.buildSearchIndex()


    def generatePerturbedPrediction(self):
        self.storage = {}

        ##### same length as original ######
        self.storage["origSrc"] = []
        self.storage["origTarg"] = []
        self.storage["origLabel"] = []
        self.storage["origPred"] = []

        ##### perturbed length ######
        # self.storage["srcSens"] = []
        # self.storage["targSens"] = []
        self.storage["mapToOrigIndex"] = []
        self.storage["pred"] = []

        correctPred = 0
        originCount = 0

        num_lines = sum(1 for line in open(self.labelFile))
        index = 0

        for _, (src_orig, targ_orig, label_orig) in \
            enumerate(itertools.izip(open(self.srcFile,'r'),
            open(self.targFile,'r'),open(self.labelFile,'r'))):
                # generate perturbation
                # print index, src_orig, targ_orig, label_orig
                label_orig = label_orig.rstrip('\n')
                targ_perb = self.perturb(targ_orig)
                src_perb = self.perturb(src_orig)

                # if self.verify(src_orig) and self.verify(targ_orig):

                self.storage["origSrc"].append(src_orig)
                self.storage["origTarg"].append(targ_orig)
                self.storage["origLabel"].append(label_orig)
                prediction = self.predict([src_orig,targ_orig])
                self.storage["origPred"].append(prediction)
                predLabel = labels[np.argmax(prediction)]

                originCount = originCount+1
                if label_orig == predLabel:
                    correctPred = correctPred+1

                ### only perturb target ####
                for targ in targ_perb:
                    # if self.verify(targ):
                            # self.storage["srcSens"].append(src_orig)
                            # self.storage["targSens"].append(targ)
                            self.storage["mapToOrigIndex"].append(index)
                            pred = self.predict([src_orig, targ])
                            # predLabel=labels[np.argmax(pred))]
                            self.storage["pred"].append(pred)

                ### only perturb src ####
                for src in src_perb:
                    # if self.verify(src):
                            # self.storage["srcSens"].append(src)
                            # self.storage["targSens"].append(targ_orig)
                            self.storage["mapToOrigIndex"].append(index)
                            pred = self.predict([src, targ_orig])
                            self.storage["pred"].append(pred)

                index = index + 1

                ####### test on small number of example #####
                # if index > 100:
                    # break
                ##### statistic #####

                # batch prediction
                if index % 20 == 0:
                    print "  processing:", str(float(index)/float(num_lines)*100.0), str(index)
                    print "current accuracy:", float(correctPred)/float(originCount)
                    # print(item, flush=True)

        # summary prediction deviation
        print "accuracy:", float(correctPred)/float(originCount)
        with open(self.saveFileName, 'wb') as handle:
            pickle.dump(self.storage, handle, protocol=pickle.HIGHEST_PROTOCOL)


##################################################################

def test_hiddenStateRecorder(filename):
    #load
    hiddenStore = hiddenStateRecorder(filename)
    neighbors = hiddenStore.neighborLookup("senEncoding", 'The woman is young .\n')
    print "reference:", 'The woman is young .\n'
    print "neighbors:", neighbors

def main(args):
    ## test neighbor lookup if states are recorded
    # test_hiddenStateRecorder('../data/test-set-hidden.pkl')
    # exit()

    #### model ####
    model = modelInterface(
        wordDict="../data/snli_1.0/snli_1.0.word.dict",
        wordVec="../data/glove.hdf5", model="../data/local_300_parikh")

    #sentence perturbation
    gen = sentenceGenerator()

    ###### test set ######
    evaluator = batchEvaluation("../data/snli_1.0/src-test.txt",
                           "../data/snli_1.0/targ-test.txt",
                           "../data/snli_1.0/label-test.txt",
                           "../data/test-pred-statistic.pkl" )

    ###### dev set ######
    # evaluator = batchEvaluation("../data/snli_1.0/src-dev.txt",
    #                        "../data/snli_1.0/targ-dev.txt",
    #                        "../data/snli_1.0/label-dev.txt",
    #                        "../data/dev-pred-statistic.pkl" )

    evaluator.setPredictionHook(model.predict)
    evaluator.setAttentionHook(model.attention)
    evaluator.setSentencePerturbationHook(gen.perturbSentence)
    # evaluator.setSentenceVerifyHook(gen.verifySentence)

    # evaluator.initialize()
    # print "finish load pkl ..."
    # evaluator.generateStatistic('../data/dev-set-statistic.json')
    # evaluator.generateStatistic('../data/test-set-statistic.json')
    ## store bson for the hidden encoding

    evaluator.generateHiddenStates()
    evaluator.saveHiddenEncoding('../data/test-set-hidden.pkl')

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
