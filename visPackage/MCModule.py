from visModule import *

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
{
    "index": 0,
    "src": "A reusable launch system (RLS, or reusable launch vehicle, RLV) is a launch system which is capable of launching a payload into space more than once. This contrasts with expendable launch systems, where each launch vehicle is launched once and then discarded. No completely reusable orbital launch system has ever been created. Two partially reusable launch systems were developed, the Space Shuttle and Falcon 9. The Space Shuttle was partially reusable: the orbiter (which included the Space Shuttle main engines and the Orbital Maneuvering System engines), and the two solid rocket boosters were reused after several months of refitting work for each launch. The external tank was discarded after each flight. \n",
    "targ": "How many partially reusable launch systems were developed?",
    "pred": "Two",
    "name":"launch system"
}
]

#############  machine comprehension ##############
class MCModule(visModule):
    def __init__(self, componentLayout):
        super(MCModule, self).__init__(componentLayout)

    def initSetup(self):
        dataManager.setData("sentenceList", exampleData)
        # dataManager.setData("pipeline", pipelineState)
        dataManager.setData("currentPair", {
            "sentences":[tokenizer(exampleData[0]['src']), tokenizer(exampleData[0]['targ'])],
            "label":exampleData[0]['pred'],
            "name": exampleData[0]['name']
        })

    def setPredictionHook(self, callback):
        self.predictionHook = callback

    def setAttentionHook(self, callback):
        self.attentionHook = callback

    def setReloadModelCallback(self, callback):
        self.reloadModelCallback = callback

    def predict(self):
        sentencePair = dataManager.getData("currentPair")['sentences']
        # print sentencePair
        predictionResult = self.predictionHook(sentencePair)
        dataManager.setData("prediction", predictionResult)
        #use raw attention
        attentionMatrix = self.attentionHook("score1")
        # print attentionMatrix
        dataManager.setData("attention", attentionMatrix)
        return True

    def predictAll(self):

        allTargetSens = None
        sentencePair = dataManager.getData("currentPair")['sentences']
        source = sentencePair[0]
        if dataManager.getData("allTargetSens") is not None:
            allTargetSens = dataManager.getData("allTargetSens")
        else:
            allTargetSens = [sentencePair[1]]
        # print "original s, t:"
        # print "all sens length:", len(allTargetSens)

        ###### if there is only one pair #####
        if len(allTargetSens) <= 1:
            return False

        allPairsPrediction = []

        for j, target in enumerate(allTargetSens):
            ######### only one perturbation is allow in each pair #######
            predResult = self.predictionHook([source, target])
            allPairsPrediction.push(predResult)
                    # allPairsPrediction[j,i,:] = predResult
        # print allPairsPrediction\
        dataManager.setData("allPairsPrediction", allPairsPrediction)
        # dataManager.setData("allAttention", allAttention)
        return True

    def reloadModel(self):
        self.reloadModelCallback();
        self.predict()
        self.predictAll()
        return True
