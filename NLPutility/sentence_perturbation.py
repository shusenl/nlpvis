import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from pattern.en import pluralize, singularize
import copy

lemmatizer = WordNetLemmatizer()

def perturb_noun_in_sentence(s, train_tokens):
	lemma_map = {}
	pos_tags = nltk.pos_tag(nltk.word_tokenize(s))
	nouns = [i if i[1].startswith("NN") else None for i in pos_tags]
	## a list of pairs (lemma, POS)
	noun_lemma = []
	for n in nouns:
		if n is not None:
			word = n[0]
			pos = n[1]
			## nltk's lemmatizer failed to make plural into singular
			## 	so make it manually
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
				## only add synonyms that has a single token (i.e. 	exclude '_')
				lemma_map[lemma].extend([ln.lower() for ln in s.lemma_names() if '_' not in ln])

				## add antonyms
				# for syn_lemma in s.lemmas():
				# 	if syn_lemma.antonyms():
				# 		lemma_map[lemma].extend([ln.name().lower() for ln in syn_lemma.antonyms()])

				## filter out words not in train dict
				lemma_map[lemma] = [l for l in lemma_map[lemma] if l in train_tokens]

				## make synonyms unique
				lemma_map[lemma] = list(set(lemma_map[lemma]))
				## remove the lemma itself
				lemma_map[lemma] = [l for l in lemma_map[lemma] if l != lemma]


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
					##	sometimes wordnet gives plural synonym when input is singular, or vice versa
					##	so need to double check
					if pos == 'NNS' and singularize(s) == s:
						target = pluralize(s)
					elif pos == "NN" and singularize(s) != s:
						target = singularize(s)

					target_list = copy.copy(orig_list)
					target_list[i] = target
					result_list.append(' '.join(target_list))

	return result_list

def perturb_noun_in_targ_file(src_path, targ_path, train_dict):
	results = []
	src = []
	targ = []
	train_tokens = {}

	with open(src_path, 'r+') as f:
		src = f.readlines()

	with open(targ_path, 'r+') as f:
		targ = f.readlines()

	# record all tokens in training set
	with open(train_dict, 'r+') as f:
		lines = f.readlines()
		for l in lines:
			toks = l.split(" ")
			train_tokens[toks[0]] = 1


	assert(len(src) == len(targ))
	print('loaded {0} sentence pairs'.format(len(src)))

	result_src = []
	result_targ = []

	for i, l in enumerate(targ):
		perturbed = perturb_noun_in_sentence(l, train_tokens)
		#add original pair for each example
		result_src.append(src[i])
		result_targ.append(l[:-1])
		for p in perturbed:
			result_src.append(src[i])
			result_targ.append(p)

		#break symbol
		result_src.append("-\n")
		result_targ.append("-")

		if (i+1)%100 == 0 or (i+1) == len(targ):
			print('perturbed {0} sentences'.format(i+1))

	src_output = '{0}.perturbed'.format(src_path)
	with open(src_output, 'w+') as f:
		for l in result_src:
			f.write('{0}'.format(l))		## no newline needed

	targ_output = '{0}.perturbed'.format(targ_path)
	with open(targ_output, 'w+') as f:
		for l in result_targ:
			f.write('{0}\n'.format(l))


if __name__ == '__main__':
	import sys
	src_path = sys.argv[1]
	targ_path = sys.argv[2]
	train_dict = sys.argv[3]
	perturb_noun_in_targ_file(src_path, targ_path, train_dict)
