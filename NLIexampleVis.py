## FIXME fix the import ###
from visPackage.nlizeModule import nlizeModule
from nli_src import modelInterface
# from NLPutility import translationPerturbation
from NLPutility import sentenceGenerator
from NLPutility import dependencyTree

#initialize NLP model
model = modelInterface(
    wordDict="data/snli_1.0/snli_1.0.word.dict",
    wordVec="data/glove.hdf5", model="data/local_300_parikh")

#sentence perturbation
# gen = translationPerturbation()
gen = sentenceGenerator()

#dependency parsing
dep = dependencyTree()

#visualization components
visLayout = {
    "column":[{"row":["Summary", "Sentence", "Pipeline"]},
            {"row":["AttentionGraph", "AttentionMatrix", "Prediction"]}]
    }

# visLayout = {
#     "column":[{"row":["Summary", "Sentence", "Prediction"]},
#             {"row":["AttentionGraph", "AttentionMatrix"]}]
#     }
#setup interface
modelVis = nlizeModule(visLayout)

modelVis.setPredictionHook(model.predict)
modelVis.setAttentionHook(model.attention)
modelVis.setPredictionUpdateHook(model.updatePrediction)
modelVis.setAttentionUpdateHook(model.updateAttention)
modelVis.setReloadModelCallback(model.reloadModel)

modelVis.setPipelineStatisticHook(model.pipelineStatistic)

modelVis.setSentencePerturbationHook(gen.perturbSentence)
modelVis.setSentenceParserHook(dep.getDependencyTree)

#open browser for visualization
# modelVis.show()
modelVis.startServer()
