# from visPackage import *
from bidaf_src import bidafModelInterface
from nli_src import modelInterface
# from sentenceGenerator import *
from NLPutility import translationPerturbation

#initialize NLP model
# trans = translationPerturbation('../key/Paraphrasing-684a368e96ad.json')

# sens = trans.perturbSentence(u"How many bird is on the large tree?")

# model = bidafModelInterface(wordDict="data/bidaf/squad.word.dict", wordVec="data/bidaf/glove.hdf5", model="data/bidaf/bidaf_5.ema")
model = bidafModelInterface("data/bidaf/squad.word.dict", "data/bidaf/glove.hdf5", "data/bidaf/bidaf_5.ema")
# model = modelInterface(wordDict="data/snli_1.0/snli_1.0.word.dict",
#         wordVec="data/glove.hdf5", model="data/local_300_parikh")
model.predict([
    "A reusable launch system (RLS, or reusable launch vehicle, RLV) is a launch system which is capable of launching a payload into space more than once. This contrasts with expendable launch systems, where each launch vehicle is launched once and then discarded. No completely reusable orbital launch system has ever been created. Two partially reusable launch systems were developed, the Space Shuttle and Falcon 9. The Space Shuttle was partially reusable: the orbiter (which included the Space Shuttle main engines and the Orbital Maneuvering System engines), and the two solid rocket boosters were reused after several months of refitting work for each launch. The external tank was discarded after each flight.",
    "How many partially reusable launch systems were developed?"
])

# pair = ["Two women are embracing while holding to go packages .\n",
# "The sisters are hugging goodbye while holding to go packages after just eating lunch .\n"]

# model.updatePrediction(pair, 0, 4, True, True, False)
