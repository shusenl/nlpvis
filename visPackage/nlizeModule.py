from .visModule import *
import pickle, sys
from NLPutility import hiddenStateRecorder
# sys.modules['hiddenStateRecorder'] = hiddenStateRecorder

############## specialized vis modules ################

'''
data organization structure
    - sentenceList (list of example pairs)
    - currentPair (the current selected pair)
    - allSourceSens (all source sentences including the oringal)
    - allTargetSens (all target sentences including the oringal)
    - allPairsPrediction (a matrix record the predict for all combination of pairs)
    - perturbedSource
    - perturbedTarget
    - prediction (the current prediction)
    - predictionsHighlight (use the index to update currentPair display)
'''

exampleData = [
{"index":0,
    "src": "<s> A couple is taking a break from bicycling .\n",
    "targ":"<s> a couple sit next to their bikes .\n",
    "pred":"neutral"
},
{
    "index":1,
    "src": "<s> Two women are embracing while holding to go packages .\n",
    "targ": "<s> The sisters are hugging goodbye while holding to go packages after just eating lunch .\n",
    "pred": "neutral"
},{
    "index":2,
    "src": "<s> Two young children in blue jerseys , one with the number 9 and one with the number 2 are standing on wooden steps in a bathroom and washing their hands in a sink .\n",
    "targ": "<s> Two kids in numbered jerseys wash their hands .\n",
    "pred": "entailment"
},{
    "index":3,
    "src": "<s> This church choir sings to the masses as they sing joyous songs from the book at a church .\n",
    "targ": "<s> The church has cracks in the ceiling .\n",
    "pred": "neutral"
},{
    "index":4,
    "src": "<s> A woman with a green headscarf , blue shirt and a very big grin .\n",
    "targ": "<s> The woman is young .\n",
    "pred": "neutral"
},{
    "index":5,
    "src": "<s> A very young child in a red plaid coat and pink winter hat makes a snowball in a large pile of snow .\n",
    "targ": "<s> A child in a red plaid coat and pink winter hat makes a snowball in a large heap of snow .\n",
    "pred": "entailment"
},{
    "index":6,
    "src": "<s> A couple is taking a break from bicycling .\n",
    "targ": "<s> sisters sit next to their bikes .\n",
    "pred": "neutral"
},{
    "index":7,
    "src": "<s> A woman in a green jacket is drinking tea .\n",
    "targ": "<s> A woman is drinking green tea .\n",
    "pred": "neutral"
}
]

#############  Natural Language Inference ##############
class nlizeModule(visModule):
    def __init__(self, componentLayout):
        super(nlizeModule, self).__init__(componentLayout)
        # self.hiddenStore = hiddenStateRecorder("data/test-set-hidden.pkl")
        self.hiddenStore = hiddenStateRecorder()
        # self.hiddenStore = None
        self.layerHook = None

    #### temp ####
    def latentStateLookup(self, sentence):
        # if sentence.startswith(u"<s>"):
        #     sentence = sentence[4:].encode('ascii','ignore')
        # if not sentence.endswith("\n"):
        #     sentence = sentence + "\n"

        neighbors = self.hiddenStore.neighborLookup("senEncoding",sentence);
        print ("reference:", sentence)
        print ("neighbors:", neighbors)
        return neighbors

    def initSetup(self):
        dataManager.setData("sentenceList", exampleData)
        dataManager.setData("pipeline", pipelineState)
        # print {"sentences":[exampleData[0]['src'], exampleData[0]['targ']],"label":exampleData[0]['pred']}
        dataManager.setData("currentPair", {"sentences":[exampleData[0]['src'], exampleData[0]['targ']],"label":exampleData[0]['pred']})

    def loadSummaryStatistic(self, filename):
        with open(filename) as json_data:
            statistics = json.load(json_data)
            # print "loadSummaryStatistic: ", type(statistics), type(statistics[0])
            dataManager.setData("evaluationStatistics", statistics)
            return True

    # an sentence pair index (self.index) is used as handle for the correspondence
    # between attention, prediction, and the input
    ### !!!! this is not called curretnly !!!! ####
    def setSentenceExample(self, data):
        sentenceList = []
        for pair in data:
            # dataManager.setData("predictions", self.data[self.index]['pred']);
            # dataManager.setData("predictionsHighlight", 0);
            # sentence = dict()
            # sentence['index'] = pair['index']
            # sentence['src'] = pair['src']
            # sentence['targ'] = pair['targ']
            sentenceList.append(pair)
        dataManager.setData("sentenceList", sentenceList)
        dataManager.setData("originalPair", [data[0]['src'], data[0]['targ']])
        dataManager.setData("currentPair", {"sentences":[data[0]['src'], data[0]['targ']],"label":data[0]['pred']})
        return True

    # called when the user change the prediction, the attention need to be
    # recomputed by python model
    def setGradientUpdateHook(self, callback):
        self.gradientUpdateHook = callback

    def setPredictionHook(self, callback):
        self.predictionHook = callback

    def setAttentionHook(self, callback):
        self.attentionHook = callback

    # access the layer activation value
    def setLayerHook(self, callback):
        self.layerHook = callback

    def setPredictionUpdateHook(self, callback):
        self.predictionUpdateHook = callback

    def setAttentionUpdateHook(self, callback):
        self.attentionUpdateHook = callback

    def setPipelineStatisticHook(self, callback):
        self.pipelineStatisticCallback = callback

    def setReloadModelCallback(self, callback):
        self.reloadModelCallback = callback

    def predictUpdate(self, newLabel, iteration, learningRate, encoderFlag, attFlag, classFlag, mira_c ):
        print ("predictUpdate", newLabel, iteration, learningRate, encoderFlag, attFlag, classFlag)
        mode = dataManager.getData("updateMode")
        sentencePair = dataManager.getData("currentPair")['sentences']
        if sentencePair[0] and sentencePair[1]:
            print (" ===== predict update mode: ", mode)
            if mode == "single":

                att, pred = self.predictionUpdateHook(sentencePair, newLabel, iteration, learningRate, encoderFlag, attFlag, classFlag)
                # print att, pred

                dataManager.setData("attention", att)
                dataManager.setData("predictionUpdate", pred)

                #update other predictions
                self.predictAll()
                self.pipelineStatistic()
                return True

            elif mode == "batch":
                self.batchRecords = []
                self.batchPreds = []
                for encoderFlag in [True, False]:
                    for attFlag in [True, False]:
                        for classFlag in [True, False]:
                            if encoderFlag | attFlag | classFlag:
                                print ("batch run:", encoderFlag, attFlag, classFlag)
                                # restore the pipeline
                                self.reloadModelCallback()
                                att, pred = self.predictionUpdateHook(sentencePair, newLabel, iteration, learningRate, encoderFlag, attFlag, classFlag, mira_c)

                                self.batchPreds.append(pred.tolist())
                                pipelineData = self.pipelineStatisticCallback()
                                self.batchRecords.append( ([encoderFlag, attFlag, classFlag], att, pipelineData) )

                dataManager.setData("predictionBatchUpdate", self.batchPreds);

        return True

    def updatePipelineStateFromIndex(self, index):
        pipeline = dataManager.getData("pipeline")
        # print pipeline
        pipeFlagList, att, pipelineData = self.batchRecords[index]
        for index, component in enumerate(pipeline):
            pipeline[index]["hist"] = pipelineData[index]
            pipeline[index]["state"] = pipeFlagList[index]

        dataManager.setData("attention", att)
        dataManager.setData("pipeline", pipeline)
        return True

    def predict(self):
        sentencePair = dataManager.getData("currentPair")['sentences']
        predictionResult = None

        predictionResult = self.predictionHook(sentencePair)

        dataManager.setData("prediction", predictionResult)
        #use raw attention
        attentionMatrix = self.attentionHook("score1")

        # print attentionMatrix
        dataManager.setData("attention", attentionMatrix)
        return True

    def attentionUpdate(self, att_soft1, att_soft2):
        sentencePair = dataManager.getData("currentPair")['sentences']
        # print sentencePair
        pred = self.attentionUpdateHook(sentencePair, att_soft1, att_soft2)
        # print pred
        dataManager.setData("prediction", pred)
        return True

    def attention(self):
        sentencePair = dataManager.getData("currentPair")['sentences']
        predictionResult = self.predictionHook(sentencePair)
        # dataManager.setData("prediction", predictionResult)
        #use raw attention
        attentionMatrix = self.attentionHook("score1")
        dataManager.setData("attention", attentionMatrix)
        return True

    def predictAll(self):

        if self.hiddenStore:
            self.hiddenStore.clear()

        allSourceSens = None
        allTargetSens = None
        sentencePair = dataManager.getData("currentPair")['sentences']
        if dataManager.getData("allSourceSens") is not None:
            allSourceSens = dataManager.getData("allSourceSens")
        else:
            allSourceSens = [sentencePair[0]]
        if dataManager.getData("allTargetSens") is not None:
            allTargetSens = dataManager.getData("allTargetSens")
        else:
            allTargetSens = [sentencePair[1]]
        # print "original s, t:"
        print ("all sens length:", len(allSourceSens), len(allTargetSens))

        ###### if there is only one pair #####
        if len(allSourceSens) <= 1 and len(allTargetSens) <= 1:
            return

        allPairsPrediction = np.zeros( (len(allSourceSens), len(allTargetSens), 3) )
        # allAttention = [None]
        wrongPred = 0
        allPred = 0
        labels = ["entailment", "neutral", "contradiction"]
        groundTruth = dataManager.getData("currentPair")['label']

        for i, source in enumerate(allSourceSens):
            for j, target in enumerate(allTargetSens):
                ######### only one perturbation is allow in each pair #######
                if i==0 or j==0:
                    predResult = self.predictionHook([source, target])
                    isCorrect = True
                    if groundTruth != labels[np.argmax(predResult)]:
                        wrongPred = wrongPred + 1
                        isCorrect = False
                    allPred = allPred + 1
                    allPairsPrediction[i,j,:] = predResult

                    # self.hiddenStore.saveTagState("senEncoding", source, self.layerHook("flat_phi1"))
                    #### TODO FIXME #### only target sentence are stored
                    if self.hiddenStore:
                        self.hiddenStore.saveDictEntry(target, isCorrect)
                    if self.hiddenStore and self.layerHook:
                        self.hiddenStore.saveTagState("senEncoding", target, self.layerHook("flat_phi2"))

                    # allPairsPrediction[j,i,:] = predResult
        # print allPairsPrediction
        print ("##### ratio:", 1.0-float(wrongPred)/float(allPred), wrongPred, allPred)
        # dataManager.setData("allAttention", allAttention)
        if self.hiddenStore:
            self.hiddenStore.buildSearchIndex("senEncoding")

        dataManager.setData("allPairsPrediction", allPairsPrediction)
        return True

    def reloadModel(self):
        self.reloadModelCallback();
        self.predict()
        self.predictAll()
        return True

    def pipelineStatistic(self):
        pipelineData = self.pipelineStatisticCallback()
        pipeline = dataManager.getData("pipeline")
        # print pipeline
        for index, component in enumerate(pipeline):
            pipeline[index]["hist"] = pipelineData[index]

        dataManager.setData("pipeline", pipeline)
        return True
