import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py
import numpy as np
import sys
import os
import random

def touch(fname, times=None):
	dir_path = os.path.dirname(os.path.abspath(fname))
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)

def load_data(data_file):
	src_sents = []
	with open(data_file, 'r+') as f:
		for l in f:
			if l.strip():
				src_sents.append(l)
	return src_sents

def load_att(att_file):
	f = h5py.File(att_file, 'r')
	return f

def sample_att(att, key):
	if key in att:
		return np.array(att[key])
	else:
		return None

def extents(f):
	delta = f[1] - f[0]
	return [f[0] - delta/2, f[-1] + delta/2]

def visualize(args):
	src_data = load_data(args.srcfile)
	targ_data = load_data(args.targfile)
	att_data = load_att(args.attfile)

	ex_idx = args.ex_idx.split(',') if args.ex_idx != 'all' else [str(i) for i in xrange(0, len(att_data))]

	for ex in ex_idx:
		ex_att = sample_att(att_data, ex)
	
		if ex_att is None:
			print("ex_idx {0} was not found in att_file.".format(ex))
			return
		else:
			print('printing attention for ex: {0}'.format(ex))

		ex_id = int(ex)
		x_tick_labels = ['<s>'] + targ_data[ex_id].split()
		y_tick_labels = ['<s>'] + src_data[ex_id].split()

		tile_height = 70
		tile_width = 100
		dpi = 100
		max_sent_l = 101
		min_size = 600 / dpi
		fig_size = (tile_width * len(x_tick_labels) / dpi , tile_height * len(y_tick_labels) / dpi)
		fig_size = (max(min_size, fig_size[0]), max(min_size, fig_size[1]))
		# all figs in the same row
		if len(ex_att.shape) == 2:
			ex_att = np.expand_dims(ex_att, axis=0)
			#ex_att.reshape((1, ex_att.shape[0], ex_att.shape[1]))
		num_labels = ex_att.shape[0]
		fig_size = (fig_size[0] * num_labels, fig_size[1])

		fig, axes = plt.subplots(figsize=fig_size, nrows=1, ncols=num_labels)

		#rand_row = random.randrange(0, ex_att.shape[1])
		#print(ex_att[:,rand_row,:].sum())

		if num_labels == 1:
			axes = [axes]

		for l in xrange(num_labels):
			ax = axes[l]

			im = ax.imshow(ex_att[l], interpolation='none')
			ax.locator_params(axis='y', nticks=len(y_tick_labels), tight=False)
			ax.locator_params(axis='x', nticks=len(x_tick_labels), tight=False)
			ax.set_xticks([i for i in range(0, len(x_tick_labels))])
			ax.set_yticks([i for i in range(0, len(y_tick_labels))])
			ax.set_xticklabels(x_tick_labels)
			ax.set_yticklabels(y_tick_labels)
	
			if args.annot == 'value':
				for y in range(ex_att[l].shape[0]):
					for x in range(ex_att[l].shape[1]):
						ax.text(x, y, '%.4f' % ex_att[l][y, x],horizontalalignment='center', verticalalignment	='center')
	
			for tick in ax.get_xticklabels():
				tick.set_rotation(70)

		fig.colorbar(im)
		fig.tight_layout()
	
		# touch file
		path = '{0}/{1}.png'.format(args.output, ex_id)
		touch(path)

		fig.savefig(path)
		plt.close('all')

		#ax = sns.heatmap(ex_att, xticklabels=x_tick_labels, yticklabels=y_tick_labels, cmap='viridis', #annot=True)
		#ax.set_xticklabels(ax.get_xticklabels(), rotation=70)
		#path = '{0}/{1}.png'.format(args.output, ex_id)
		#ax.figure.savefig(path)
		#ax.figure.tight_layout()
		#plt.close('all')



def main(arguments):
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--srcfile', help="Path to sent1 data.", default = "../data/snli_1.0/src-eval.txt")
	parser.add_argument('--targfile', help="Path to sent2 data.", default = "../data/snli_1.0/targ-eval.txt")
	parser.add_argument('--attfile', help="Path to attention data.", default = "")
	parser.add_argument('--annot', help="To annotate with cell value, set to <value>", default = "none")
	parser.add_argument('--ex_idx', help="Example id to visualize (starting from 1), separated by ','", default = "all")
	parser.add_argument('--output', help="Path to an output folder", default = "./att_output/")
	args = parser.parse_args(arguments)
	visualize(args)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
