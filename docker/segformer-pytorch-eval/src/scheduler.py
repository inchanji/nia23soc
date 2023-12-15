import numpy as np
import torch
import math
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR, LambdaLR
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupRestarts(_LRScheduler):
	"""
		optimizer (Optimizer): Wrapped optimizer.
		first_cycle_steps (int): First cycle step size.
		cycle_mult(float): Cycle steps magnification. Default: -1.
		max_lr(float): First cycle's max learning rate. Default: 0.1.
		min_lr(float): Min learning rate. Default: 0.001.
		warmup_steps(int): Linear warmup step size. Default: 0.
		gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
		last_epoch (int): The index of last epoch. Default: -1.
	"""
	
	def __init__(self,
				 optimizer : torch.optim.Optimizer,
				 first_cycle_steps : int,
				 cycle_mult : float = 1.,
				 max_lr : float = 0.1,
				 min_lr : float = 0.001,
				 warmup_steps : int = 0,
				 gamma : float = 1.,
				 last_epoch : int = -1
		):
		assert warmup_steps < first_cycle_steps
		
		self.first_cycle_steps = first_cycle_steps # first cycle step size
		self.cycle_mult = cycle_mult # cycle steps magnification
		self.base_max_lr = max_lr # first max learning rate
		self.max_lr = max_lr # max learning rate in the current cycle
		self.min_lr = min_lr # min learning rate
		self.warmup_steps = warmup_steps # warmup step size
		self.gamma = gamma # decrease rate of max learning rate by cycle
		
		self.cur_cycle_steps = first_cycle_steps # first cycle step size
		self.cycle = 0 # cycle count
		self.step_in_cycle = last_epoch # step size of the current cycle
		self.curEpoch = 0    
		super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
		
		# set learning rate min_lr
		self.init_lr()
	
	def init_lr(self):
		self.base_lrs = []
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.min_lr
			self.base_lrs.append(self.min_lr)
	
	def get_last_lr(self):
		idx = self.curEpoch % len(self.get_lr())
		return [self.get_lr()[idx]]

	def get_lr(self):
		if self.step_in_cycle == -1:
			return self.base_lrs
		elif self.step_in_cycle < self.warmup_steps:
			return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
		else:
			return [base_lr + (self.max_lr - base_lr) \
					* (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
									/ (self.cur_cycle_steps - self.warmup_steps))) / 2
					for base_lr in self.base_lrs]

	def step(self, epoch=None):
		self.curEpoch += 1
		if epoch is None:
			epoch = self.last_epoch + 1
			self.step_in_cycle = self.step_in_cycle + 1
			if self.step_in_cycle >= self.cur_cycle_steps:
				self.cycle += 1
				self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
				self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
		else:
			if epoch >= self.first_cycle_steps:
				if self.cycle_mult == 1.:
					self.step_in_cycle = epoch % self.first_cycle_steps
					self.cycle = epoch // self.first_cycle_steps
				else:
					n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
					self.cycle = n
					self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
					self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
			else:
				self.cur_cycle_steps = self.first_cycle_steps
				self.step_in_cycle = epoch
				
		self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
		self.last_epoch = math.floor(epoch)
		for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
			param_group['lr'] = lr



def get_scheduler(config, optimizer, steps_per_epoch = 1000):
	if config.scheduler=='ReduceLROnPlateau':
		scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.factor, patience=config.patience, verbose=True, eps=config.eps)
	elif config.scheduler=='CosineAnnealingLR':
		scheduler = CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.min_lr, last_epoch=-1)
	elif config.scheduler=='CosineAnnealingWarmRestarts':
		scheduler = CosineAnnealingWarmupRestarts(optimizer,
										  first_cycle_steps = config.T_0,
										  cycle_mult 		= 1.0,
										  max_lr 			= config.lr,
										  min_lr 			= config.min_lr,
										  warmup_steps 		= 5,
										  gamma 			= 0.5
										  )
	elif config.scheduler=='OneCycleLR':
		scheduler = OneCycleLR(optimizer, pct_start=0.1, div_factor=100, max_lr=5.0e-5, epochs=config.epochs, steps_per_epoch=steps_per_epoch)# steps_per_epoch = len(train_loader))
	elif config.scheduler=='DecayingOscillation':
		lf = lambda x: config.min_lr + (config.lr - config.min_lr) * math.exp(-config.tau*x/config.osc_t) * (1 + math.cos(x * math.pi / config.osc_t)) / 2
		scheduler = LambdaLR(optimizer, lr_lambda=lf)
	elif config.scheduler=='DecayingOscillation_v2':
		lf = lambda x: config.min_lr + (config.lr - config.min_lr) * math.exp(-(np.sign(x-10)+1)*0.5 *config.tau*x/config.osc_t) * (1. + math.cos( 0.3 * (np.sign(x-10)+1)*0.5 * x * (x/config.osc_t + 1) * math.pi / config.osc_t)) / 2.
		scheduler = LambdaLR(optimizer, lr_lambda=lf)
	else: 
		scheduler = None
	return scheduler