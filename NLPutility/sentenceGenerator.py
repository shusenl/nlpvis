import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from pattern.en import pluralize, singularize
import copy

import sys
sys.path.insert(0, '..')

class sentenceGenerator:
    def __init__(self, train_dict='data/snli_1.0/snli_1.0.word.dict'):
    # def __init__(self, train_dict='../data/snli_1.0/snli_1.0.word.dict'):
        self.train_tokens = {}
        with open(train_dict, 'r+') as f:
            lines = f.readlines()
            for l in lines:
                toks = l.split()
                self.train_tokens[toks[0]] = 1
        ### perturb one sentence to preload the nltk ####
        # self.perturbSentence("Who am I.")
        # print "angele:", "angele" in self.train_tokens.keys()
        # print "token count:", len(self.train_tokens.keys())

    def verifySentence(self, sen):
        for word in sen.split():
            if word not in self.train_tokens.keys():
                return False
        return True

    #perturb nouns and verb in the sentence
    def perturbSentence(self, inputSentence):
        train_tokens = self.train_tokens
        lemma_map = {}
        pos_tags = nltk.pos_tag(nltk.word_tokenize(inputSentence))
        nouns = [i if i[1].startswith("NN") else None for i in pos_tags]
        ## a list of pairs (lemma, POS)
        noun_lemma = []
        for n in nouns:
            if n is not None:
                word = n[0]
                pos = n[1]
                ## nltk's lemmatizer failed to make plural into singular
                ##     so make it manually
                if pos == 'NNS':
                    noun_lemma.append((singularize(word), pos))
                else:
                    noun_lemma.append((word, pos))
            else:
                noun_lemma.append(None)

        for l in noun_lemma:
            if l is not None:
                lemma = l[0]
                synsets = wn.synsets(lemma)
                if len(synsets) != 0:
                    lemma_map[lemma] = []

                for s in synsets:
                    ## add synonyms
                    ## only add synonyms that has a single token (i.e.     exclude '_')
                    lemma_map[lemma].extend([ln for ln in s.lemma_names() if '_' not in ln])

                    ## add antonyms
                    # for syn_lemma in s.lemmas():
                    #     if syn_lemma.antonyms():
                    #         lemma_map[lemma].extend([ln.name().lower() for ln in syn_lemma.antonyms()])

                    ## filter out words not in train dict
                    lemma_map[lemma] = [l for l in lemma_map[lemma] if l in train_tokens.keys()]

                    ## make synonyms unique
                    lemma_map[lemma] = list(set(lemma_map[lemma]))
                    ## remove the lemma itself
                    lemma_map[lemma] = [l for l in lemma_map[lemma] if l != lemma]
        # print lemma_map

        orig_list = [i[0] for i in pos_tags]
        result_list = []
        for i, l in enumerate(noun_lemma):
            if l is not None:
                lemma = l[0]
                pos = l[1]
                if lemma in lemma_map:
                    synonyms = lemma_map[lemma]
                    for s in synonyms:
                        target = s
                        ## dealing with plural and sigular
                        ##    sometimes wordnet gives plural synonym when input is singular, or vice versa
                        ##    so need to double check
                        if pos == 'NNS' and singularize(s) == s:
                            target = pluralize(s)
                        elif pos == "NN" and singularize(s) != s:
                            target = singularize(s)

                        #if the plural or singular form does not exists in the training then continue
                        if target not in train_tokens.keys():
                            continue

                        target_list = copy.copy(orig_list)
                        target_list[i] = target
                        result_list.append(' '.join(target_list))

        return result_list
