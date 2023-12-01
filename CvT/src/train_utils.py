import os 
import time
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from src.loss import f1_score, f1_score_numpy
from src.dataset import classInfo


def train_one_epoch(epoch, 
					config, 
					model, 
					loss_fn, 
					optimizer, 
					train_loader, 
					device, 
					scheduler=None, 
					schd_batch_update=False, 
					wandb = None
					):
	if config.apex:
		scaler = GradScaler()

	running_loss 	= None
	sample_num 		= 0
	loss_sum 		= 0
	metric_sum 		= 0

	classinfo 		 = classInfo(include_normal = config.include_normal)
	# confusion_matrix = torch.zeros(len(classinfo), len(classinfo))

	# switch to train mode
	model.train()

	pbar = tqdm(enumerate(train_loader), total=len(train_loader))
	for step, (_, imgs, image_labels) in pbar:
		imgs = imgs.to(device).float()

		if config.do_fmix or config.do_cutmix or config.BCELoss:
			image_labels = image_labels.to(device).float()
		else:
			image_labels = image_labels.to(device).long()

		if config.apex:
			with autocast():
				image_preds = model(imgs)
				loss 		= loss_fn(image_preds.view(-1) if config.BCELoss else image_preds, image_labels)
		else:
			image_preds = model(imgs)
			loss 		= loss_fn(image_preds.view(-1) if config.BCELoss else image_preds, image_labels)
			
		if config.accum_iter > 1:
			loss = loss / config.accum_iter

		if config.apex:
			scaler.scale(loss).backward()
		else:
			loss.backward()
	
		if running_loss is None:
			running_loss = loss.item()
		else:
			running_loss = running_loss * .99 + loss.item() * .01 

		loss_sum 	+= loss.item()*image_labels.shape[0]*config.accum_iter
		sample_num 	+= image_labels.shape[0] * config.accum_iter

		if ((step + 1) %  config.accum_iter == 0) or ((step + 1) == len(train_loader)):
			# may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
			if config.apex:
				scaler.step(optimizer)
				scaler.update()
			else:
				optimizer.step()
			optimizer.zero_grad()

			if scheduler is not None and schd_batch_update:
				if config.scheduler == 'ReduceLROnPlateau':
					scheduler.step(running_loss)
				else:
					scheduler.step()
		
		# apply softmax to output
		image_preds = torch.sigmoid(image_preds)
		metric      = f1_score(image_labels, image_preds, multiclass = config.multiclass)
		metric_sum += metric*image_labels.shape[0]
		

		description = f'epoch {epoch} loss: {running_loss:.4f}'
		pbar.set_description(description)			

		if wandb is not None:	
			wandb.log({ f"train_loss": loss_sum/sample_num,
						f"f1_score": metric,
					    f"lr": scheduler.get_last_lr()[0]})
			
		if config.debug and (step > 16):
			break					
			
	if scheduler is not None and not schd_batch_update:
		if config.scheduler == 'ReduceLROnPlateau':
			scheduler.step(loss_sum/sample_num)
		else:
			scheduler.step()


	return loss_sum/sample_num, metric_sum/sample_num



def valid_one_epoch(epoch, 
					config, 
					model, 
					loss_fn, 
					val_loader, 
					device, 
					threshold = 0.5, 
					scheduler=None, 
					schd_loss_update=False, 
					wandb = None, 
					oof = None
					):
	model.eval()

	classinfo 		 = classInfo(include_normal = config.include_normal)
	dir2save 		= f"outputs/{config.expName}"

	loss_sum 	= 0
	sample_num 	= 0

	image_preds_all 	= []
	image_targets_all 	= []
	image_onehot_targets_all = []
	image_conf_all 		= []
	image_pred_conf_all = []
	fnames 				= []

	if config.multiclass:
		threshold = torch.tensor([threshold]*config.num_classes).to(device)

	pbar = tqdm(enumerate(val_loader), total=len(val_loader))
	for step, (fname, image, image_labels) in pbar:
		with torch.no_grad():
			image_preds   = model(image.float().to(device))
			
		# loss 		= loss_fn(image_preds, image_labels.to(device))
		loss 		= loss_fn(image_preds.view(-1) if config.num_classes == 1 else image_preds, image_labels.to(device))

		loss_sum 	+= loss.item()*image_labels.shape[0]
		sample_num 	+= image_labels.shape[0]

		if ((step + 1) % config.verbose_step == 0) or ((step + 1) == len(val_loader)):
			description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
			pbar.set_description(description)		

			if config.multiclass:
				# make pred as one-hot-encoding
				image_preds_all 	+= [torch.ge(torch.sigmoid(image_preds), threshold).float().detach().cpu().numpy()]
				image_pred_conf_all += [torch.sigmoid(image_preds).detach().cpu().numpy()]
			else:
				image_preds_all 	+= [torch.argmax(image_preds, 1).detach().cpu().numpy()]
				image_pred_conf_all	+= [torch.softmax(image_preds, 1).detach().cpu().numpy()]
			
			image_conf_all 		+= [image_preds.detach().cpu().numpy()]

		image_onehot_targets_all += [image_labels.long().detach().cpu().numpy()]
		image_targets_all 	+= [torch.argmax(image_labels, 1).long().detach().cpu().numpy()]
		fnames 				+= [fname]

		if config.debug and (step > 16):
			break		

	image_preds_all 	= np.concatenate(image_preds_all)
	image_targets_all 	= np.concatenate(image_targets_all)
	image_conf_all 		= np.concatenate(image_conf_all)
	image_onehot_targets_all = np.concatenate(image_onehot_targets_all)
	fnames_all			= np.concatenate(fnames)

	if not config.num_classes == 1: 
		image_pred_conf_all = np.concatenate(image_pred_conf_all)

	val_accuracy 		= np.mean(image_preds_all.ravel()==image_targets_all)
	val_f1_score 		= f1_score_numpy(image_onehot_targets_all, image_pred_conf_all, multiclass = config.multiclass)

	# confusion matrix to wandb
	if wandb is not None:
		wandb.log({ f"confusion_matrix_{oof}": wandb.plot.confusion_matrix(probs=None,
																	 y_true=image_targets_all.astype(int),
																	 preds=image_preds_all.astype(int),
																	 class_names=classinfo.get_class_names(),
																	 title=f"Confusion matrix {oof}")})

	
		
			
	if oof is not None:
		df = pd.DataFrame({
			'id': fnames_all,
			'target': image_targets_all.astype(int),
			'pred': image_preds_all,
			'conf': image_pred_conf_all.max(1),
		})

		if not os.path.exists(f"{dir2save}/oof"):
			os.makedirs(f"{dir2save}/oof")

		df.to_csv(f"{dir2save}/oof/{oof}.csv", index=False)

	return loss_sum/sample_num, val_accuracy, val_f1_score

