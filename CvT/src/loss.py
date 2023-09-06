import torch
from torch import nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import math
import numpy as np
# from .config import CFG
from scipy.special import lambertw



class FocalLoss(nn.Module):
	"""
	The focal loss for fighting against class-imbalance
	"""

	def __init__(self, alpha=1, gamma=2):
		super(FocalLoss, self).__init__()
		self.alpha 		= alpha
		self.gamma 		= gamma
		self.epsilon 	= 1e-12  # prevent training from Nan-loss error

	def forward(self, logits, target):
		"""
		logits & target should be tensors with shape [batch_size, num_classes]
		"""
		probs 				= F.sigmoid(logits)
		one_subtract_probs 	= 1.0 - probs

		# add epsilon
		probs_new 				= probs + self.epsilon
		one_subtract_probs_new 	= one_subtract_probs + self.epsilon

		# calculate focal loss
		log_pt 		= target * torch.log(probs_new) + (1.0 - target) * torch.log(one_subtract_probs_new)
		pt 			= torch.exp(log_pt)
		focal_loss 	= -1.0 * (self.alpha * (1 - pt) ** self.gamma) * log_pt

		return torch.mean(focal_loss)



class SuperLoss(nn.Module):
	def __init__(self, config, C=10, lam=2, valid = False):
		super(SuperLoss, self).__init__()
		self.tau 		= math.log(C)
		self.lam 		= lam  # set to 1 for CIFAR10 and 0.25 for CIFAR100
		self.valid 		= valid
		self.config 	= config
		
		if valid:
			self.batchsize = self.config.train_bs
		else:
			self.batchsize = self.config.valid_bs

	
	def forward(self, logits, targets):
		#l_i = F.cross_entropy(logits, targets, reduction='none').detach()
		l_i = self.cross_entropy(logits, targets).detach()
		#print('l_i:{}'.format(l_i))
		sigma = self.sigma(l_i)
		#loss = (F.cross_entropy(logits, targets, reduction='none') - self.tau)*sigma + self.lam*(torch.log(sigma)**2)
		loss = (self.cross_entropy(logits, targets) - self.tau)*sigma + self.lam*(torch.log(sigma)**2)
		loss = loss.sum()/self.batchsize
		return loss

	def sigma(self, l_i):
		x 		= torch.ones(l_i.size())*(-2/math.exp(1.))
		x 		= x.to(self.config.device)
		y 		= 0.5*torch.max(x, (l_i-self.tau)/self.lam)
		y 		= y.cpu().numpy()
		sigma 	= np.exp(-lambertw(y))
		sigma 	= sigma.real.astype(np.float32)
		sigma 	= torch.from_numpy(sigma).to(self.config.device)
		return sigma

	def cross_entropy(self, logits, targets):
		if self.valid:
			return F.cross_entropy(logits, targets, reduction='none')

		if self.config.do_fmix or self.config.do_cutmix:
			lsm 	= F.log_softmax(logits, -1) 
			loss 	= -(targets * lsm).sum(-1)
			return loss
		else:
			return F.cross_entropy(logits, targets, reduction='none')


# reference: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/173733
class MyCrossEntropyLoss(_WeightedLoss):
	def __init__(self, weight = None, reduction = 'mean'):
		super().__init__(weight = weight, reduction = reduction)
		self.weight 	= weight
		self.reduction = reduction

	def forward(self, inputs, targets):
		lsm = F.log_softmax(inputs, -1)

		if self.weight is not None:
			lsm = lsm * self.weight.unsqueeze(0)

		loss = -(targets * lsm).sum(-1)
		if  self.reduction == 'sum':
			loss = loss.sum()
		elif  self.reduction == 'mean':
			loss = loss.mean()
		return loss                    


class MyCrossEntropyLossOof(_WeightedLoss):
	def __init__(self, weight=None, reduction='mean'):
		super().__init__(weight=weight, reduction=reduction)
		self.weight = weight
		self.reduction = reduction

	def forward(self, inputs, targets, weight_oof = None):
		lsm = F.log_softmax(inputs, -1)

		if self.weight is not None:
			lsm = lsm * self.weight.unsqueeze(0)

		# print("lsm:",lsm)
		# print("targets:",targets)

		loss = -(targets * lsm).sum(-1)

		if  self.reduction == 'sum':
			loss = loss.sum()
		elif  self.reduction == 'mean':
			loss = loss.mean()
		return loss   


def get_loss_fn(config, device, valid = False, weight = None):
	if valid:
		return nn.CrossEntropyLoss().to(device)

	if config.loss == 'floss': 
		return FocalLoss(alpha=1, gamma=2).to(device)
	elif config.loss == 'sloss':
		return SuperLoss(config, C=config.num_classes, lam = config.lam, valid = valid).to(device)	
	elif config.loss == 'bceloss':
		return nn.BCEWithLogitsLoss().to(device)
	elif config.loss == 'cEloss':
		return MyCrossEntropyLoss(weight = weight).to(device)



	# if config.superLoss:
	# 	loss = SuperLoss(config, C=config.num_classes, lam = config.lam, valid = valid).to(device)	
	# else:
	# 	if valid:
	# 		loss  = nn.CrossEntropyLoss().to(device)
	# 	elif (config.do_fmix or config.do_cutmix or config.rectpackmix):
	# 		loss  = MyCrossEntropyLoss(weight = weight).to(device) # if one-hot-encoding
	# 	elif config.BCELoss:
	# 		loss = nn.BCEWithLogitsLoss().to(device)
	# 	else:
	# 		loss  = MyCrossEntropyLoss(weight = weight).to(device)
	# 		# loss  = nn.CrossEntropyLoss().to(device) 

	# return loss


# def get_loss_fn_oof(device, valid = False):
# 	if CFG.superLoss:
# 		loss = SuperLoss(C=CFG.num_classes, lam = CFG.lam, batch_size = CFG.train_bs, valid = valid).to(device)	
# 	else:
# 		if CFG.BCELoss:
# 			loss = nn.BCEWithLogitsLoss(reduction='none').to(device)
# 		else:
# 			loss  = nn.CrossEntropyLoss(reduction='none').to(device) 
# 	return loss


# torch.log  and math.log is e based
class WingLoss(nn.Module):
	def __init__(self, omega=10, epsilon=2, fac = 1):
		super(WingLoss, self).__init__()
		self.omega 		= omega
		self.epsilon 	= epsilon
		self.fac 		= fac

	def forward(self, pred, target):
		y 	 	 = target
		y_hat 	 = torch.softmax(pred, 1)[:,1].view(-1)
		delta_y  = self.fac * (y - y_hat).abs()  # delta_y: 0 - 10
		
		delta_y1 = delta_y[delta_y < self.omega]
		delta_y2 = delta_y[delta_y >= self.omega]
		loss1 	 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
		C 		 = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
		loss2 	 = delta_y2 - C
		return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


def weightedMultiClassLogarithmicLoss(conf, target, M, weights = None):
	#  	 conf: confidence value of prediction
	#  target: label class
	# 		M: the number of classes

	# print("conf.shape", conf.shape)    # (sample size, M)
	# print("target.shape", target.shape)  # (sample size, )
	# print("Ncls", M)

	smoothing = 1e-15

	if M == 1:
		conf = np.concatenate([1-conf, conf], axis = 1)
		M += 1

	# weights
	if weights is None:
		weights = [1 for _ in range(M)]

	# pred 	= np.argmax(conf, 1).astype(int)
	target 	= target.astype(int)

	# the number of images in the class set
	Ntot    = len(target)
	Wi  	= np.array([ weights[target[i]] for i in range(Ntot) ])

	Cij 	= np.array([np.max([np.min([conf[i, target[i]],1.-smoothing]), smoothing]) for i in range(Ntot) ])
	# print("Cij:", Cij)
	lnPij 	= np.array([ np.log(Cij[i]) for i in range(Ntot) ])

	logLoss = -1. * np.sum( Wi  * lnPij ) / np.sum(Wi)

	return logLoss




# f1 score 
def f1_score(y_true, y_pred, threshold=0.5):
	return fbeta_score(y_true, y_pred, 1, threshold)

def fbeta_score(y_true, y_pred, beta, threshold, eps=1e-9):
	beta2 = beta ** 2

	y_pred = torch.ge(y_pred.float(), threshold).float()
	y_true = y_true.float()

	true_positive = (y_pred * y_true).sum(dim=1)
	precision = true_positive.div(y_pred.sum(dim=1).add(eps))
	recall = true_positive.div(y_true.sum(dim=1).add(eps))

	return torch.mean( precision.mul(recall).div(precision.mul(beta2) + recall + eps).mul(1 + beta2))

# numpy version of f1 score
def f1_score_numpy(y_true, y_pred, threshold=0.5):
	return fbeta_score_numpy(y_true, y_pred, 1, threshold)

def fbeta_score_numpy(y_true, y_pred, beta, threshold, eps=1e-9):
	beta2 = beta ** 2

	y_pred = np.greater_equal(y_pred.astype(np.float32), threshold).astype(np.float32)
	y_true = y_true.astype(np.float32)

	true_positive = np.sum(y_pred * y_true, axis=1)
	precision = true_positive / (np.sum(y_pred, axis=1) + eps)
	recall = true_positive / (np.sum(y_true, axis=1) + eps)

	return np.mean(precision * recall / (precision * beta2 + recall + eps) * (1 + beta2))