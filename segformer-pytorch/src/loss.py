# loss functions for semantic segmentation

import torch
from torch import nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
import math
import numpy as np

# binary segmentation loss

# class DiceLoss(nn.Module):
# 	def __init__(self, weight=None, size_average=True):
# 		super(DiceLoss, self).__init__()

# 	def forward(self, inputs, targets, smooth=1):
		
# 		#comment out if your model contains a sigmoid or equivalent activation layer
# 		inputs = F.sigmoid(inputs)       
		
# 		#flatten label and prediction tensors
# 		inputs = inputs.view(-1)
# 		targets = targets.view(-1)
		
# 		intersection = (inputs * targets).sum()                            
# 		dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
		
# 		return 1 - dice


def dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
	# Average of Dice coefficient for all batches, or for a single mask
	assert input.size() == target.size()
	assert input.dim() == 3 or not reduce_batch_first

	sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

	inter = 2 * (input * target).sum(dim=sum_dim)
	sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
	sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

	dice = (inter + epsilon) / (sets_sum + epsilon)
	return dice.mean()


def multiclass_dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
	# Average of Dice coefficient for all classes
	return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


# def dice_loss(input: torch.Tensor, target: torch.Tensor, multiclass: bool = False):
#     # Dice loss (objective to minimize) between 0 and 1
#     fn = multiclass_dice_coeff if multiclass else dice_coeff
#     return 1 - fn(input, target, reduce_batch_first=True)



class DiceLoss(nn.Module):
	def __init__(self, reduce_batch_first = False, epsilon = 1e-6):
		super(DiceLoss, self).__init__()
		self.reduce_batch_first = reduce_batch_first
		self.epsilon = epsilon

	def forward(self, inputs, targets):
		# Average of Dice coefficient for all batches, or for a single mask
		# print(inputs.size(), targets.size())
		inputs = inputs.flatten(0, 2)   # (N, H, W,C) -> (N*H*W, C)
		targets = targets.flatten(0, 2) # (N, H, W,C) -> (N*H*W, C)

		if inputs.size() != targets.size():
			print(inputs.size(), targets.size())
			assert inputs.size() == targets.size()
		assert inputs.dim() == 3 or not self.reduce_batch_first
		
		sum_dim = (-1, -2) if inputs.dim() == 2 or not self.reduce_batch_first else (-1, -2, -3)

		inter = 2 * (inputs * targets).sum(dim=sum_dim)
		sets_sum = inputs.sum(dim=sum_dim) + targets.sum(dim=sum_dim)
		sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

		dice = (inter + self.epsilon) / (sets_sum + self.epsilon)
		return 1 - dice.mean()



# Jaccard loss
class IoULoss(nn.Module):
	def __init__(self, weight=None, size_average=True):
		super(IoULoss, self).__init__()

	def forward(self, inputs, targets, smooth=1):
		
		#comment out if your model contains a sigmoid or equivalent activation layer
		inputs = F.sigmoid(inputs)       
		
		#flatten label and prediction tensors
		inputs = inputs.view(-1)
		targets = targets.view(-1)
		
		#intersection is equivalent to True Positive count
		#union is the mutually inclusive area of all labels & predictions 
		intersection = (inputs * targets).sum()
		total = (inputs + targets).sum()
		union = total - intersection 
		
		IoU = (intersection + smooth)/(union + smooth)
				
		return 1 - IoU
	



class FocalLoss(nn.Module):
	def __init__(self, weight=None, size_average=True, alpha=0.8, gamma=2):
		super(FocalLoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma

	def forward(self, inputs, targets, smooth=1, alpha=None, gamma=None):
		if alpha is None:
			alpha = self.alpha
		if gamma is None:
			gamma = self.gamma
		
		#comment out if your model contains a sigmoid or equivalent activation layer
		inputs = F.sigmoid(inputs)       
		
		#flatten label and prediction tensors
		inputs = inputs.view(-1)
		targets = targets.view(-1)
		
		#first compute binary cross-entropy 
		BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
		BCE_EXP = torch.exp(-BCE)
		focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
					   
		return focal_loss


class TverskyLoss(nn.Module):
	def __init__(self, weight=None, size_average=True, alpha=0.8, beta=2):
		super(TverskyLoss, self).__init__()
		self.alpha = alpha
		self.beta = beta

	def forward(self, inputs, targets, smooth=1, alpha=None, beta=None):
		if alpha is None:
			alpha = self.alpha
		if beta is None:
			beta = self.beta
		
		#comment out if your model contains a sigmoid or equivalent activation layer
		inputs = F.sigmoid(inputs)       
		
		#flatten label and prediction tensors
		inputs = inputs.view(-1)
		targets = targets.view(-1)
		
		#True Positives, False Positives & False Negatives
		TP = (inputs * targets).sum()    
		FP = ((1-targets) * inputs).sum()
		FN = (targets * (1-inputs)).sum()
	   
		Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
		
		return 1 - Tversky



class FocalTverskyLoss(nn.Module):
	def __init__(self, weight=None, size_average=True, alpha= 0.5, beta=0.5, gamma=1):
		super(FocalTverskyLoss, self).__init__()
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma

	def forward(self, inputs, targets, smooth=1, alpha=None, beta=None, gamma=None):
		if alpha is None:
			alpha = self.alpha
		if beta is None:
			beta = self.beta
		if gamma is None:
			gamma = self.gamma
		
		#comment out if your model contains a sigmoid or equivalent activation layer
		inputs = F.sigmoid(inputs)       
		
		#flatten label and prediction tensors
		inputs = inputs.view(-1)
		targets = targets.view(-1)
		
		#True Positives, False Positives & False Negatives
		TP = (inputs * targets).sum()    
		FP = ((1-targets) * inputs).sum()
		FN = (targets * (1-inputs)).sum()
		
		Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
		FocalTversky = (1 - Tversky)**gamma
					   
		return FocalTversky


class ComboLoss(nn.Module):
	def __init__(self, weight=None, size_average=True, alpha=0.5, ce_ratio=0.5):
		super(ComboLoss, self).__init__()
		self.alpha = alpha
		self.ce_ratio = ce_ratio

	def forward(self, inputs, targets, smooth=1, eps=1e-9, alpha=None, ce_ratio=None):
		if alpha is None:
			alpha = self.alpha
		if ce_ratio is None:
			ce_ratio = self.ce_ratio

		#flatten label and prediction tensors
		inputs = inputs.view(-1)
		targets = targets.view(-1)
		
		#True Positives, False Positives & False Negatives
		intersection = (inputs * targets).sum()    
		dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
		
		inputs = torch.clamp(inputs, eps, 1.0 - eps)       
		out = - (alpha * ((targets * torch.log(inputs)) + ((1 - alpha) * (1.0 - targets) * torch.log(1.0 - inputs))))
		weighted_ce = out.mean(-1)
		combo = (ce_ratio * weighted_ce) - ((1 - ce_ratio) * dice)
		
		return combo



# --------------------------- MULTICLASS LOSSES ---------------------------
# mean IOU 

# Multi-class Dice loss 
class DiceLossMulti(torch.nn.Module):
	def __init__(self, weight=None, size_average=True):
		super(DiceLossMulti, self).__init__()
		

		
	def forward(self, 
				inputs : torch.Tensor, # (N, H, W, num_classes)
				targets: torch.Tensor, # (N, H, W)
				smooth=1) -> torch.Tensor : # (N, H, W) 
		
		# argmax along the num_classes dimension
		inputs = torch.argmax(inputs, dim=2)

		
		#flatten label and prediction tensors
		inputs = inputs.view(-1)
		targets = targets.view(-1)

		# print(inputs.shape)
		# print(targets.shape)
		
		intersection = (inputs * targets).sum()                            
		dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
		
		return 1 - dice
	 
def mIoU(pred_mask, mask, smooth=1e-10, n_classes=23):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)






class IoU(torch.nn.Module):
	def __init__(self, weight=None, size_average=True):
		super(IoU, self).__init__()

	def forward(self, inputs, targets, smooth=1):
		
		#comment out if your model contains a sigmoid or equivalent activation layer
		# inputs = F.sigmoid(inputs)       
		
		#flatten label and prediction tensors
		inputs = inputs.view(-1)
		targets = targets.view(-1)
		
		#intersection is equivalent to True Positive count
		#union is the mutually inclusive area of all labels & predictions 
		intersection = (inputs * targets).sum()
		total = (inputs + targets).sum()
		union = total - intersection 
		
		IoU = (intersection + smooth)/(union + smooth)
				
		return IoU
	

# reference: https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/173733
class MyCrossEntropyLoss(_WeightedLoss):
	def __init__(self, weight = None, reduction = 'mean'):
		super().__init__(weight = weight, reduction = reduction)
		self.weight 	= weight
		self.reduction 	= reduction

	def forward(self, inputs, targets):
		# inputs shape: (B, C, H, W)
		# targets shape: (B, H, W)

		# permute inputs to shape (B, H, W, C) and then flatten to (B*H*W, C)
		# flatten targets to (B*H*W)
		inputs 	= inputs.permute(0, 2, 3, 1).flatten(0, 2)
		targets = targets.flatten(0,2)

		# make targets to one-hot (B*H*W, C)
		targets = F.one_hot(targets, inputs.shape[1]).float()

		lsm = F.log_softmax(inputs, dim=1)

		# multiply weight along the last dimension
		if self.weight is not None:
			lsm = lsm * self.weight.unsqueeze(0)

		loss = -(targets * lsm).sum(-1)

		if self.reduction == 'mean':
			return loss.mean()
		elif self.reduction == 'sum':
			return loss.sum()


def get_loss_fn(config, device, valid = False, weight = None):
	loss = {}
	loss['loss_iou']  = DiceLoss().to(device)
	loss['loss_cls']  = MyCrossEntropyLoss(weight = weight).to(device)

	return loss

	# if config.multiclass:
	# 	# multi-label classification
	# 	return DiceLossMulti().to(device)
	# 	# return MultiLabelCrossEntropyLoss(weight = weight).to(device)
	# else:
	# 	return DiceLossMulti().to(device)




if __name__ == '__main__':
	
	batch 		= 2
	imgsize 	= 4
	num_classes = 3
	pred 	= torch.randn(batch, num_classes, imgsize, imgsize)
	target 	= torch.randint(0, num_classes, (batch, imgsize, imgsize))

	print(pred)
	print(target)

	pred_i 	= torch.argmax(pred, 1)

	print(pred_i)

	print((pred_i == target).sum())

						   
	bincount = torch.bincount( pred_i.reshape(-1).long() * num_classes + target.long().reshape(-1), 
										minlength = num_classes**2
										).reshape(num_classes, num_classes)	

	print(bincount)

	normalized_bincount = bincount / (target.reshape(-1).shape[0]) 
	print(normalized_bincount)


	iou = bincount.diag() / (bincount.sum(dim=1) + bincount.sum(dim=0) - bincount.diag())
	iou_norm = normalized_bincount.diag() / (normalized_bincount.sum(dim=1) + normalized_bincount.sum(dim=0) - normalized_bincount.diag())
	print(iou)
	print(iou_norm)



	lsm = F.log_softmax(pred, dim=1)

	lsm = lsm.permute(0, 2, 3, 1).flatten(0, 2)
	target = target.flatten(0,2)
	# target to one-hot
	target = F.one_hot(target, num_classes).float()
	


	print(lsm.shape)
	print(target.shape)

	loss = -(target * lsm).sum(-1)
	print(loss)

	weight = torch.Tensor([0.0, 1.0, 1.0])
	print(target)
	
	# multiply weight along the last dimension
	weighted_target = target * weight.unsqueeze(0)
	print(weighted_target)
	# loss_w = loss.mean()