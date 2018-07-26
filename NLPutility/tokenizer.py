import nltk

def tokenizer(sentence):
    sentence = nltk.word_tokenize(sentence)
    sentence = [t.replace("''", '"').replace("``", '"') for t in sentence]
    #
    return " ".join(sentence)
    # print "token: ", sentence
