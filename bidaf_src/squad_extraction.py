import ujson
import sys
import argparse
import nltk
import re

DEBUG = False

def get_gold(answer_spans):
	cnt = {}
	for span in answer_spans:
		if span in cnt:
			cnt[span] = cnt[span] + 1
		else:
			cnt[span] = 1
	sorted_keys = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
	maj_span = sorted_keys[0][0]
	return (maj_span, answer_spans.index(maj_span))


def write_to(ls, out_file):
	print('writing to {0}'.format(out_file))
	with open(out_file, 'w+') as f:
		for l in ls:
			f.write((l + '\n').encode('utf-8'))


def remap_char_idx(context, context_toks):
	context_tok_seq = ' '.join(context_toks)
	m = [-1 for _ in range(len(context))]
	i = 0
	j = 0
	while (i < len(context) and j < len(context_tok_seq)):
		# skip white spaces
		while context[i].strip() == '':
			i += 1
		while context_tok_seq[j].strip() == '':
			j += 1

		if context[i] == context_tok_seq[j]:
			m[i] = j
			i += 1
			j += 1
		elif context[i] == "'" and context[i+1] == "'" and context_tok_seq[j] == '"':
			m[i] = j
			i += 2
			j += 1
		#elif context[i] == '"' and context_tok_seq[j] == '\'':
		#	m[i] = j
		#	i += 1
		#	if context_tok_seq[j+1] == '\'':
		#		j += 2
		else:
			print(context.encode('utf8'))
			print(context_tok_seq.encode('utf8'))
			print(context[:i+1].encode('utf8'))
			print(context_tok_seq[:j+1].encode('utf8'))
			assert(False)

	return m


def map_answer_idx(context, context_toks, char_idx1, char_idx2):
	context_tok_seq = ' '.join(context_toks)
	remap = remap_char_idx(context, context_toks)
	new_char_idx1 = remap[char_idx1]
	new_char_idx2 = remap[char_idx2]

	# count number of spaces
	tok_idx1 = context_tok_seq[new_char_idx1::-1].count(' ')
	tok_idx2 = context_tok_seq[new_char_idx2::-1].count(' ')

	# sanity check
	assert(tok_idx1 < len(context_toks))
	assert(tok_idx2 < len(context_toks))

	# NOTE, ending index is inclusive
	return (tok_idx1, tok_idx2)


def extract(json_file):
	all_context = []
	all_query = []
	all_span = []
	context_max_sent_num = 0
	max_sent_l = 0

	with open(json_file, 'r') as f:
		f_str = f.read()
	j_obj = ujson.loads(f_str)

	data = j_obj['data']

	for article in data:
		title = article['title']
		pars = article['paragraphs']
		for p in pars:
			context = p['context']
			qas = p['qas']
			# tokenize
			context_toks = nltk.word_tokenize(context)
			context_toks = [t.replace("''", '"').replace("``", '"') for t in context_toks]

			for qa in qas:
				query = qa['question']
				ans = qa['answers']
				# tokenize
				query_toks = nltk.word_tokenize(query)
				query_toks = [t.replace("''", '"').replace("``", '"') for t in query_toks]

				max_sent_l = max(max_sent_l, len(query_toks))

				answer_orig_spans = []
				for a in ans:
					a_txt = a['text']
					idx1 = a['answer_start']
					idx2 = idx1 + len(a_txt) - 1	# end idx is inclusive

					answer_orig_spans.append((idx1, idx2))

				orig_maj_span = get_gold(answer_orig_spans)[0]
				# map orig char idx to tokenized word idx
				tok_idx1, tok_idx2 = map_answer_idx(context, context_toks, orig_maj_span[0], orig_maj_span[1])

				# sanity check
				orig_answer = context[orig_maj_span[0]:orig_maj_span[1]+1]
				matched_answer = context_toks[tok_idx1:tok_idx2+1]
				print((idx1, idx2), (tok_idx1, tok_idx2), orig_answer, matched_answer)

				# add to final list
				all_context.append(' '.join(context_toks))
				all_query.append(' '.join(query_toks))
				all_span.append((tok_idx1, tok_idx2))

		print('max sent len: {0}'.format(max_sent_l))

	return (all_context, all_query, all_span)


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data', help="Path to SQUAD json file", default="data/dev-v1.1.json")
parser.add_argument('--output', help="Prefix to the path of output", default="data/dev")


def main(args):
	opt = parser.parse_args(args)
	context, query, span = extract(opt.data)
	print('{0} examples processed.'.format(len(context)))

	write_to(context, opt.output + '.context.txt')
	write_to(query, opt.output + '.query.txt')

	span = ['{0} {1}'.format(p[0], p[1]) for p in span]
	write_to(span, opt.output + '.span.txt')



if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))


