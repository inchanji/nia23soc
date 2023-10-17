import setproctitle
import pandas as pd
import platform
import torch

def get_full_path(path, root = ""):
	return "{}/{}".format(root, path)	

def display_cfg(config):
	print('model arch    : {}'.format(config.model_arch))
	print('img size      : {}'.format(config.imgsize))
	print('batch size(tr): {}'.format(config.train_bs))
	print('scheduler     : {}'.format(config.scheduler))
	print('lr            : {}'.format(config.lr))	



def get_modelname_ext(config):
	ext = ''
	ext += '{}px'.format(config.imgsize)

	ext += '-bs{}'.format(config.train_bs)


	if config.scheduler: 
		if config.scheduler == 'CosineAnnealingLR': 
			ext += '-cosanneal'
		elif config.scheduler == 'ReduceLROnPlateau':
			ext += '-reduceplateau'
		elif config.scheduler == 'CosineAnnealingWarmRestarts':
			ext += '-cosannealwarm'
		elif config.scheduler == 'DecayingOscillation':
			ext += '-decayosc'
		elif config.scheduler == 'DecayingOscillation_v2':
			ext += '-decayosc2'

	ext += "-" + config.loss

	if config.is_grayscale:
		ext += "-grayimg" 
	else:
		ext += "-3chimg" 

	if config.valid_tta: 
		ext += "-tta" 		

	if config.multiclass: 
		ext += "-mc"
	else:
		ext += "-sc"

	return ext


def select_device(device):
	if platform.system() == 'Linux':
		device 	= torch.device(device)
	elif platform.system() == 'Darwin':
		if torch.backends.mps.is_available() and device == 'mps': 
			device = torch.device("mps")
		else:  
			device = torch.device("cpu")
	else: # windows
		device 	= torch.device("cpu")
	return device


def class2dict(f):
	return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

def set_proc_name(config, newname):
	setproctitle.setproctitle(newname + "-" + get_modelname_ext(config))