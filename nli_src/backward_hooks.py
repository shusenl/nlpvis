
# defines some callback functions registered in pipeline
#	registered functions will be called during forward pass
#	for instance, add one of the functions to pipeline during eval.py running
import torch
from torch import nn
from torch import cuda
from holder import *
from torch.autograd import Variable
import numpy as np
import warnings
import math


def register_grad(shared, name):
    def hook(g):
        shared.grad[name] = g
    return hook


def retain_grad(shared, v_name):
	if hasattr(shared, v_name):
		v = getattr(shared, v_name)
		v.retain_grad()
	else:
		print('shared holder has no variable {0} recorded'.format(v_name))


