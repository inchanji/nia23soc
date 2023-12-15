import torch

def prepare_optimizer(config, model):
	if config.optimizer == 'adam':
		return torch.optim.Adam(model.parameters(), lr = config.lr, weight_decay = config.weight_decay)
	elif config.optimizer == 'rmsOp':
		return torch.optim.RMSprop(model.parameters(), lr = config.lr, weight_decay = config.weight_decay, momentum = 0)
	else:
		return None