import os 
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from src.transforms import MEANPIXVAL, STDPIXVAL
import torch.nn.functional as F
import cv2
# from src.dataset import classID, classID_inc_normal
from src.dataset import classInfo
# from matplotlib import pyplot as plt
from src.plot_utils import plot_confusion_matrix
import torch.distributed as dist


    
def undo_preprocess(img, imgsize = (128, 128), mean=MEANPIXVAL*255, std=STDPIXVAL*255):
	img = (img*std + mean).astype(np.uint8)
	return cv2.resize(img, imgsize) 


def train_one_epoch(epoch, 
					config, 
					model, 
					loss_fn, 
					train_loader, 					
					optimizer, 
					device, 
					scheduler 		= None, 
					schd_batch_update=False, 
					visualize 		= True,
					max_uploads 	 = 64, 
					taskname 		 = 'train',
					wandb = None
					):
	
	classinfo = classInfo(include_normal = config.include_normal)

	dir2save = f"outputs/{config.expName}"

	# switch to train mode
	model.train()
	if config.apex:
		scaler = GradScaler()	
	
	running_loss = None
	sample_num 		= 0
	loss_sum 		= 0

	images_wandb 	= []
	masks_wandb  	= []	

	# num_classes 	 = config.num_classes + 1 if config.include_normal else config.num_classes
	# class_names 	 = classID_inc_normal if config.include_normal else classID
	class_names 	 = classinfo.get_class_names()
	num_classes 	 = classinfo.num_classes

	# pytorch version of confusion matrix
	normalized_confusion_matrix = torch.zeros(num_classes, num_classes).to(device)

	pbar = tqdm(enumerate(train_loader), total=len(train_loader))

	for step, (images, labels, image_path) in pbar:
		images 	= images.to(device).float()
		labels 	= labels.to(device).float()
		batch_size = labels.shape[0]
		sample_num += batch_size
		
		if config.apex:
			with autocast():
				preds 	= model(images.float().to(device)).logits
		else:
			preds = model(images.float().to(device)).logits

		# resize preds to match labels
		# pred shape: (B, C, H, W)
		# label shape: (B, H, W)

		loss_cls = loss_fn['loss_cls'](preds, labels.long())
		loss_iou = loss_fn['loss_iou'](F.softmax(preds, dim = 1).permute(0,2,3,1).float(),  # pred: (B, C, H, W) to (B, H, W, C)
							  		F.one_hot(labels.long(), num_classes).float() # label: (B, H, W) to (B, H, W, C)
									)
		
		loss = loss_cls + loss_iou 
		
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

		loss_sum += loss.item()*labels.shape[0]*config.accum_iter
		sample_num += labels.shape[0] * config.accum_iter

		# make confusion matrix for miou
		preds_i 	= torch.argmax(preds, 1)	# (B, C, H, W) to (B, H, W)
		labels_i 	= labels					# (B, H, W)

		# update confusion matrix
		binount = torch.bincount( preds_i.reshape(-1).long() * num_classes + labels_i.long().reshape(-1), 
											minlength = num_classes**2
											).reshape(num_classes, num_classes)
		normalized_confusion_matrix_i = binount / (labels_i.reshape(-1).shape[0]) 
		normalized_confusion_matrix += normalized_confusion_matrix_i


		iou_i = []
		for i in range(num_classes):
			iou = normalized_confusion_matrix_i[i,i] / (normalized_confusion_matrix_i[i,:].sum() + normalized_confusion_matrix_i[:,i].sum() - normalized_confusion_matrix_i[i,i])
			iou_i.append(iou.item())

		# replace nan with 0 
		iou_i = np.nan_to_num(iou_i)

		if ((step + 1) %  config.accum_iter == 0) or ((step + 1) == len(train_loader)):
			description = f'epoch {epoch} loss({taskname}, cls/iou): {loss_cls.item():.4f}, {loss_iou.item():.4f}'
			pbar.set_description(description)

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

		if visualize and (wandb is not None) and (len(images_wandb) < max_uploads):
			batch_size  	= images.shape[0]
			imgsize 		= labels.shape[1:3]

			_images = images.detach().cpu().numpy().reshape([batch_size, -1, images.shape[2],images.shape[3]])
			_images = np.transpose(_images, (0,2,3,1))
			_images = [undo_preprocess(_images[i,:,:,:], imgsize = imgsize)  for i in range(batch_size)]

			_labels  = labels.detach().cpu().numpy().reshape([batch_size, labels.shape[1], labels.shape[2]]).astype(np.uint8)

			# mask non-zero labels on _images
			signals = [	_images[i] *  np.repeat(_labels[i,:,:,np.newaxis] > 0, 3,-1) for i in range(batch_size)]

			# visualize gradients of preds 
			# print(preds.shape)
			preds_w = preds.softmax(1).permute(0,2,3,1).detach().cpu().numpy()     # (B, H, W, C)
			preds_i = np.argmax(preds.detach().cpu().numpy(), 1).astype(np.uint8)  # (B, H, W)

			# print(preds_w.shape)
			# print(preds_i.shape)	

			for i in range(batch_size):
				images_wandb.append(wandb.Image(_images[i], caption = "{}".format(image_path[i].split('/')[-1])))
				masks_wandb.append( wandb.Image( signals[i].astype(np.uint8), caption = "{}".format(image_path[i].split('/')[-1])))

				if len(images_wandb) >= max_uploads:
					break

		if wandb is not None:
			wandb.log({	f"{taskname}_loss_cls": loss_cls.item(),
						f"{taskname}_loss_iou": loss_iou.item(),
						f"{taskname}_loss": loss.item(),
						f"{taskname}_lr": scheduler.get_last_lr()[0]
						})

			for i in range(num_classes):
				wandb.log({	f"{taskname}_iou_{class_names[i]}": iou_i[i]})

		if config.debug and (step > 16):
			break

	if not os.path.exists(f"{dir2save}/plots"):
		os.makedirs(f"{dir2save}/plots", exist_ok=True)

	# confusion matrix including normal
	normalized_confusion_matrix = normalized_confusion_matrix.detach().cpu().numpy().transpose()
	normalized_confusion_matrix /= normalized_confusion_matrix.sum()

	# plot confusion matrix
	img_confusion_mat = plot_confusion_matrix(	normalized_confusion_matrix, 
										   		num_classes, class_names, 
												f"{dir2save}/plots/{taskname}_confusion_matrix.png", 
												taskname)

	# confusion matrix excluding normal
	confusion_matrix = normalized_confusion_matrix[1:,1:]
	confusion_matrix /= confusion_matrix.sum()
	img_confusion_mat_excluding_normal = plot_confusion_matrix( confusion_matrix, 
																num_classes-1, 
																class_names[1:], 
																f"{dir2save}/plots/{taskname}_confusion_matrix_excluding_normal.png", 
																taskname)

	if wandb is not None:
		wandb.log({	f"{taskname}_confusion_matrix": 				 wandb.Image(img_confusion_mat, caption = f"{taskname}_confusion_matrix"),
					f"{taskname}_confusion_matrix_excluding_normal": wandb.Image(img_confusion_mat_excluding_normal, caption = f"{taskname}_confusion_matrix_excluding_normal")
					})

	if visualize and (wandb is not None):
			wandb.log({	f"{taskname}_images": images_wandb,
						f"{taskname}_masks" : masks_wandb})
			

	# calculate iou and miou using normalized_confusion_matrix
	iou_all = []
	for i in range(num_classes):
		iou_all.append(normalized_confusion_matrix[i,i] / (normalized_confusion_matrix[i,:].sum() + normalized_confusion_matrix[:,i].sum() - normalized_confusion_matrix[i,i]))
		
	miou_all = np.nanmean(iou_all)
	miou_wo_normal = np.nanmean(iou_all[1:])

	if wandb is not None:
		wandb.log({	f"{taskname}_miou_avg(w/ bg)": miou_all})	
		wandb.log({	f"{taskname}_miou_avg(no bg)": miou_wo_normal})
		for i in range(num_classes):
			wandb.log({	f"{taskname}_iou_{class_names[i]}_avg": iou_all[i]})
		wandb.log({	f"{taskname}_loss_avg": loss_sum/sample_num})

	if scheduler is not None and not schd_batch_update:
		if config.scheduler == 'ReduceLROnPlateau':
			scheduler.step(loss_sum/sample_num)
		else:
			scheduler.step()

	return loss_sum/sample_num, {'mIoU_all': miou_all, 'mIoU_wo_normal': miou_wo_normal, 'iou': iou_all}


def train_one_epoch_ddp(epoch, 
					config, 
					model, 
					loss_fn, 
					train_loader, 					
					optimizer, 
					device, 
					scheduler 		= None, 
					schd_batch_update=False, 
					visualize 		= True,
					max_uploads 	 = 64, 
					taskname 		 = 'train',
					wandb = None
					):
	dist.barrier()
	classinfo = classInfo(include_normal = config.include_normal)

	dir2save = f"outputs/{config.expName}"

	# switch to train mode
	model.train()
	if config.apex:
		scaler = GradScaler()	
	
	running_loss = None
	sample_num 		= 0
	loss_sum 		= 0

	images_wandb 	= []
	masks_wandb  	= []	

	# num_classes 	 = config.num_classes + 1 if config.include_normal else config.num_classes
	# class_names 	 = classID_inc_normal if config.include_normal else classID
	class_names 	 = classinfo.get_class_names()
	num_classes 	 = classinfo.num_classes

	# pytorch version of confusion matrix
	normalized_confusion_matrix = torch.zeros(num_classes, num_classes).to(device)

	if config.rank in [-1, 0]:
		pbar = tqdm(enumerate(train_loader), total=len(train_loader))
	else:
		pbar = enumerate(train_loader)

	print(f'start training of rank {config.rank}')
	for step, (images, labels, image_path) in pbar:
		images 	= images.to(device).float()
		labels 	= labels.to(device).float()
		batch_size = labels.shape[0]
		sample_num += batch_size
		
		if config.apex:
			with autocast():
				preds 	= model(images.float().to(device)).logits
		else:
			preds = model(images.float().to(device)).logits

		# resize preds to match labels
		# pred shape: (B, C, H, W)
		# label shape: (B, H, W)

		loss_cls = loss_fn['loss_cls'](preds, labels.long())
		loss_iou = loss_fn['loss_iou'](F.softmax(preds, dim = 1).permute(0,2,3,1).float(),  # pred: (B, C, H, W) to (B, H, W, C)
							  		F.one_hot(labels.long(), num_classes).float() # label: (B, H, W) to (B, H, W, C)
									)
		
		loss = loss_cls + loss_iou 
		
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

		loss_sum += loss.item()*labels.shape[0]*config.accum_iter
		sample_num += labels.shape[0] * config.accum_iter

		# make confusion matrix for miou
		preds_i 	= torch.argmax(preds, 1)	# (B, C, H, W) to (B, H, W)
		labels_i 	= labels					# (B, H, W)

		# update confusion matrix
		binount = torch.bincount( preds_i.reshape(-1).long() * num_classes + labels_i.long().reshape(-1), 
											minlength = num_classes**2
											).reshape(num_classes, num_classes)
		normalized_confusion_matrix_i = binount / (labels_i.reshape(-1).shape[0]) 
		normalized_confusion_matrix += normalized_confusion_matrix_i


		iou_i = []
		for i in range(num_classes):
			iou = normalized_confusion_matrix_i[i,i] / (normalized_confusion_matrix_i[i,:].sum() + normalized_confusion_matrix_i[:,i].sum() - normalized_confusion_matrix_i[i,i])
			iou_i.append(iou.item())

		# replace nan with 0 
		iou_i = np.nan_to_num(iou_i)

		if ((step + 1) %  config.accum_iter == 0) or ((step + 1) == len(train_loader)):
			if config.rank in [-1, 0]:
				description = f'epoch {epoch} loss({taskname}, cls/iou): {loss_cls.item():.4f}, {loss_iou.item():.4f}'
				pbar.set_description(description)

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

		if config.rank in [-1, 0] and visualize and (wandb is not None) and (len(images_wandb) < max_uploads):
			batch_size  	= images.shape[0]
			imgsize 		= labels.shape[1:3]

			_images = images.detach().cpu().numpy().reshape([batch_size, -1, images.shape[2],images.shape[3]])
			_images = np.transpose(_images, (0,2,3,1))
			_images = [undo_preprocess(_images[i,:,:,:], imgsize = imgsize)  for i in range(batch_size)]

			_labels  = labels.detach().cpu().numpy().reshape([batch_size, labels.shape[1], labels.shape[2]]).astype(np.uint8)

			# mask non-zero labels on _images
			signals = [	_images[i] *  np.repeat(_labels[i,:,:,np.newaxis] > 0, 3,-1) for i in range(batch_size)]

			# visualize gradients of preds 
			# print(preds.shape)
			preds_w = preds.softmax(1).permute(0,2,3,1).detach().cpu().numpy()     # (B, H, W, C)
			preds_i = np.argmax(preds.detach().cpu().numpy(), 1).astype(np.uint8)  # (B, H, W)

			# print(preds_w.shape)
			# print(preds_i.shape)	

			for i in range(batch_size):
				images_wandb.append(wandb.Image(_images[i], caption = "{}".format(image_path[i].split('/')[-1])))
				masks_wandb.append( wandb.Image( signals[i].astype(np.uint8), caption = "{}".format(image_path[i].split('/')[-1])))

				if len(images_wandb) >= max_uploads:
					break

		if config.rank in [-1, 0] and wandb is not None and not config.debug:	
			wandb.log({	f"{taskname}_loss_cls": loss_cls.item(),
						f"{taskname}_loss_iou": loss_iou.item(),
						f"{taskname}_loss": loss.item(),
						f"{taskname}_lr": scheduler.get_last_lr()[0]
						})

			for i in range(num_classes):
				wandb.log({	f"{taskname}_iou_{class_names[i]}": iou_i[i]})

		if config.debug and (step > 16):
			break

	if not os.path.exists(f"{dir2save}/plots"):
		os.makedirs(f"{dir2save}/plots", exist_ok=True)

	# confusion matrix including normal
	normalized_confusion_matrix = normalized_confusion_matrix.detach().cpu().numpy().transpose()
	normalized_confusion_matrix /= normalized_confusion_matrix.sum()

	# plot confusion matrix
	if config.rank in [-1, 0] and not config.debug:
		img_confusion_mat = plot_confusion_matrix(	normalized_confusion_matrix, 
										   		num_classes, class_names, 
												f"{dir2save}/plots/{taskname}_confusion_matrix.png", 
												taskname)

	# confusion matrix excluding normal
	confusion_matrix = normalized_confusion_matrix[1:,1:]
	confusion_matrix /= confusion_matrix.sum()
	
	if config.rank in [-1, 0] and not config.debug:
		img_confusion_mat_excluding_normal = plot_confusion_matrix( confusion_matrix, 
																num_classes-1, 
																class_names[1:], 
																f"{dir2save}/plots/{taskname}_confusion_matrix_excluding_normal.png", 
																taskname)

	if config.rank in [-1, 0] and wandb is not None and not config.debug:	
		wandb.log({	f"{taskname}_confusion_matrix": 				 wandb.Image(img_confusion_mat, caption = f"{taskname}_confusion_matrix"),
					f"{taskname}_confusion_matrix_excluding_normal": wandb.Image(img_confusion_mat_excluding_normal, caption = f"{taskname}_confusion_matrix_excluding_normal")
					})

	if config.rank in [-1, 0] and visualize and wandb is not None and not config.debug:
			wandb.log({	f"{taskname}_images": images_wandb,
						f"{taskname}_masks" : masks_wandb})
			

	# calculate iou and miou using normalized_confusion_matrix
	iou_all = []
	for i in range(num_classes):
		iou_all.append(normalized_confusion_matrix[i,i] / (normalized_confusion_matrix[i,:].sum() + normalized_confusion_matrix[:,i].sum() - normalized_confusion_matrix[i,i]))
		
	miou_all = np.nanmean(iou_all)
	miou_wo_normal = np.nanmean(iou_all[1:])

	if config.rank in [-1, 0] and wandb is not None and not config.debug:	
		wandb.log({	f"{taskname}_miou_avg(w/ bg)": miou_all})	
		wandb.log({	f"{taskname}_miou_avg(no bg)": miou_wo_normal})
		for i in range(num_classes):
			wandb.log({	f"{taskname}_iou_{class_names[i]}_avg": iou_all[i]})
		wandb.log({	f"{taskname}_loss_avg": loss_sum/sample_num})

	if scheduler is not None and not schd_batch_update:
		if config.scheduler == 'ReduceLROnPlateau':
			scheduler.step(loss_sum/sample_num)
		else:
			scheduler.step()


	return loss_sum/sample_num, {'mIoU_all': miou_all, 'mIoU_wo_normal': miou_wo_normal, 'iou': iou_all}




def valid_one_epoch(epoch, 
					config, 
					model, 
					loss_fn, 
					val_loader, 
					device, 
					threshold 		 = 0.5, 
					scheduler		 = None, 
					schd_loss_update = False, 
					wandb 			 = None,
					visualize 		 = True,
					max_uploads 	 = 64, 
					taskname 		 = 'val' # 'val' or 'test', 
					):
	
	classinfo = classInfo(include_normal = config.include_normal)

	dir2save = f"outputs/{config.expName}"

	model.eval()
	loss_sum 	= 0
	sample_num 	= 1e-10
	metric 		= None

	images_wandb = []
	masks_wandb  = []
	
	upload_image_wandb 	= False

	
	# class_names 	 = classID_inc_normal if config.include_normal else classID	
	class_names 	 = classinfo.get_class_names()
	num_classes 	 = classinfo.num_classes

	normalized_confusion_matrix = torch.zeros(num_classes, num_classes).to(device)

	pbar = tqdm(enumerate(val_loader), total=len(val_loader))

	for step, (images, labels, image_path) in pbar:
		labels =  labels.to(device)
		preds   = model(images.float().to(device)).logits

		# calculate loss
		loss_cls = loss_fn['loss_cls'](preds, labels.long())
		loss_iou = loss_fn['loss_iou'](F.softmax(preds, dim = 1).permute(0,2,3,1).float(),
										F.one_hot(labels.long(), num_classes).float()
									) 

		loss = loss_cls + loss_iou
		loss_sum += loss.item()*labels.shape[0]

		sample_num += labels.shape[0]

		# make confusion matrix for miou
		preds_i 	= torch.argmax(preds, 1)	# (B, C, H, W) to (B, H, W)
		labels_i 	= labels					# (B, H, W)

		# update confusion matrix
		binount = torch.bincount( preds_i.reshape(-1).long() * num_classes + labels_i.long().reshape(-1), 
											minlength = num_classes**2
											).reshape(num_classes, num_classes)
		normalized_confusion_matrix_i = binount / (labels_i.reshape(-1).shape[0]) 
		normalized_confusion_matrix += normalized_confusion_matrix_i

		description = f'epoch {epoch} loss({taskname}, cls/iou): {loss_cls.item():.4f}, {loss_iou.item():.4f}'
		pbar.set_description(description)

		if visualize and wandb is not None and not config.debug:
			if (len(images_wandb) >= max_uploads) and (not upload_image_wandb):
				upload_image_wandb = True	
				wandb.log({	f"{taskname}_images": images_wandb,
							f"{taskname}_masks" : masks_wandb
							})			

							
		if visualize and (wandb is not None) and (len(images_wandb) < max_uploads):
			batch_size  	= images.shape[0]
			imgsize 		= labels.shape[1:3]

			_images = images.detach().cpu().numpy().reshape([batch_size, -1, images.shape[2],images.shape[3]])
			_images = np.transpose(_images, (0,2,3,1))
			_images = [undo_preprocess(_images[i,:,:,:], imgsize = imgsize)  for i in range(batch_size)]

			_labels  = labels.detach().cpu().numpy().reshape([batch_size, labels.shape[1], labels.shape[2]]).astype(np.uint8)

			# mask non-zero labels on _images
			signals = [	_images[i] *  np.repeat(_labels[i,:,:,np.newaxis] > 0, 3,-1) for i in range(batch_size)]

			# visualize gradients of preds 
			# print(preds.shape)
			preds_w = preds.softmax(1).permute(0,2,3,1).detach().cpu().numpy()     # (B, H, W, C)
			preds_i = np.argmax(preds.detach().cpu().numpy(), 1).astype(np.uint8)  # (B, H, W)

			for i in range(batch_size):
				images_wandb.append(wandb.Image(_images[i], caption = "{}".format(image_path[i].split('/')[-1])))
				masks_wandb.append(wandb.Image( signals[i].astype(np.uint8), caption = "{}".format(image_path[i].split('/')[-1])))

				if len(images_wandb) >= max_uploads:
					break


		if config.debug and (step > 16):
			break

	if not os.path.exists(f"{dir2save}/plots"):
		os.makedirs(f"{dir2save}/plots", exist_ok=True)


	# confusion matrix including normal
	normalized_confusion_matrix = normalized_confusion_matrix.detach().cpu().numpy().transpose()
	normalized_confusion_matrix /= normalized_confusion_matrix.sum()

	# plot confusion matrix
	img_confusion_mat = plot_confusion_matrix(	normalized_confusion_matrix, 
												num_classes, 
												class_names, 
												f"{dir2save}/plots/{taskname}_confusion_matrix.png", 
												taskname)

	# confusion matrix excluding normal
	confusion_matrix = normalized_confusion_matrix[1:,1:]
	confusion_matrix /= confusion_matrix.sum()
	img_confusion_mat_excluding_normal = plot_confusion_matrix(	confusion_matrix, 
																num_classes-1, 
																class_names[1:], 
																f"{dir2save}/plots/{taskname}_confusion_matrix_excluding_normal.png", 
																taskname)				

	# calculate iou and miou using normalized_confusion_matrix
	iou_all = []
	for i in range(num_classes):
		iou_all.append(normalized_confusion_matrix[i,i] / (normalized_confusion_matrix[i,:].sum() + normalized_confusion_matrix[:,i].sum() - normalized_confusion_matrix[i,i]))
	miou_all = np.nanmean(iou_all)
	miou_wo_normal = np.nanmean(iou_all[1:])

	if wandb is not None and not config.debug:
		wandb.log({	f"{taskname}_miou_avg(w/ bg)": miou_all})	
		wandb.log({	f"{taskname}_miou_avg(no bg)": miou_wo_normal})
		for i in range(num_classes):
			wandb.log({	f"{taskname}_iou_{class_names[i]}_avg": iou_all[i]})
		wandb.log({	f"{taskname}_loss_avg": loss_sum/sample_num})


	if wandb is not None and not config.debug:
		wandb.log({	f"{taskname}_confusion_matrix": 				 wandb.Image(img_confusion_mat, caption = f"{taskname}_confusion_matrix"),
					f"{taskname}_confusion_matrix_excluding_normal": wandb.Image(img_confusion_mat_excluding_normal, caption = f"{taskname}_confusion_matrix_excluding_normal")
					})
		
	return loss_sum/sample_num, {'mIoU_all': miou_all, 'mIoU_wo_normal': miou_wo_normal, 'iou': iou_all}




def evaluate(		epoch, 
					config, 
					model, 
					loss_fn, 
					val_loader, 
					device, 
					threshold 		 = 0.5, 
					scheduler		 = None, 
					schd_loss_update = False, 
					wandb 			 = None,
					visualize 		 = False,
					max_uploads 	 = 64, 
					taskname 		 = 'test' # 'val' or 'test', 
					):
	classinfo = classInfo(include_normal = config.include_normal)

	dir2save = f"outputs/{config.expName}"

	model.eval()
	loss_sum 	= 0
	sample_num 	= 1e-10
	metric 		= None

	images_wandb = []
	masks_wandb  = []
	
	upload_image_wandb 	= False

	# class_names 	 = classID_inc_normal if config.include_normal else classID	
	class_names 	 = classinfo.get_class_names()
	num_classes 	 = classinfo.num_classes

	normalized_confusion_matrix 	= torch.zeros(num_classes, num_classes).to(device)
	normalized_confusion_matrix_i 	= torch.zeros(num_classes, num_classes).to(device)

	pbar = tqdm(enumerate(val_loader), total=len(val_loader))

	iou_all 	= []
	fname 		= ''
	n_images 	= 0
	miou 		= 0
	miou_sum 	= 0

	metric 		= { 'num_classes': [0]* num_classes, 
			 		'iou_sum_per_cls': [0]* num_classes,
			 		'miou_sum': 0,
					'num_images': 1e-6,
					'miou_sum(w/o bg)': 0,
					'num_images(w/o bg)': 1e-6
					}

	for step, (images, labels, image_path) in pbar:
		labels =  labels.to(device)
		preds   = model(images.float().to(device)).logits

		# calculate loss
		if loss_fn is not None:
			loss_cls = loss_fn['loss_cls'](preds, labels.long()).item()
			loss_iou = loss_fn['loss_iou'](F.softmax(preds, dim = 1).permute(0,2,3,1).float(),
											F.one_hot(labels.long(), num_classes).float()
										).item()
	
			loss = loss_cls + loss_iou
			loss_sum += loss*labels.shape[0]
		else:
			loss_iou = -1
			loss_cls = -1
			loss_sum = -1

		sample_num += labels.shape[0]

		# make confusion matrix for miou
		preds_i 	= torch.argmax(preds, 1)	# (B, C, H, W) to (B, H, W)
		labels_i 	= labels					# (B, H, W)

		for i, path in enumerate(image_path):
			_fname 		= path.split('/')[-1]
			pred_i 		= preds_i[i, :, :]
			label_i 	= labels_i[i, :, :]

			bincount 	= torch.bincount( pred_i.reshape(-1).long() * num_classes + label_i.long().reshape(-1),
										minlength = num_classes**2
										).reshape(num_classes, num_classes)
			
			if fname != _fname:
				if fname == '':
					fname = _fname
					continue

				# n_images 			+= 1
				metric['num_images'] += 1

				iou = []
				for j in range(num_classes):
					val = normalized_confusion_matrix_i[j, j] / (normalized_confusion_matrix_i[j, :].sum() + normalized_confusion_matrix_i[:, j].sum() - normalized_confusion_matrix_i[j, j])
					iou.append(val.item())
					metric['iou_sum_per_cls'][j] += val.item() if not np.isnan(val.item()) else 0
					metric['num_classes'][j] += 1 if not np.isnan(val.item()) else 0
				

				miou_each = np.nanmean(iou)
				if not np.isnan(np.nanmean(iou[1:])):
					metric['miou_sum(w/o bg)'] 		+= np.nanmean(iou[1:])
					metric['num_images(w/o bg)'] 	+= 1

				if not np.isnan(miou_each):
					metric['miou_sum'] += miou_each
				else:
					metric['num_images'] -= 1

				miou = metric['miou_sum'] / metric['num_images']	
				miou_no_bg = metric['miou_sum(w/o bg)'] / metric['num_images(w/o bg)'] if metric['num_images(w/o bg)'] > 0 else 0


				description = f'epoch {epoch}, {fname:35s}, loss({taskname}, cls|iou): {loss_cls:.4f}|{loss_iou:.4f}, miou(each): {miou_each:.4f}, miou(avg.): {miou:.4f}'
				pbar.set_description(description)

				fname = _fname
				normalized_confusion_matrix_i[:, :] = bincount / (label_i.reshape(-1).shape[0]) 
				
			else:
				normalized_confusion_matrix_i[:, :] += bincount / (label_i.reshape(-1).shape[0]) 
				


		if visualize and (wandb is not None):
			if (len(images_wandb) >= max_uploads) and (not upload_image_wandb):
				upload_image_wandb = True	
				wandb.log({	f"{taskname}_images": images_wandb,
							f"{taskname}_masks" : masks_wandb
							})			

							
		if visualize and (wandb is not None) and (len(images_wandb) < max_uploads):
			batch_size  	= images.shape[0]
			imgsize 		= labels.shape[1:3]

			_images = images.detach().cpu().numpy().reshape([batch_size, -1, images.shape[2],images.shape[3]])
			_images = np.transpose(_images, (0,2,3,1))
			_images = [undo_preprocess(_images[i,:,:,:], imgsize = imgsize)  for i in range(batch_size)]

			_labels  = labels.detach().cpu().numpy().reshape([batch_size, labels.shape[1], labels.shape[2]]).astype(np.uint8)

			# mask non-zero labels on _images
			signals = [	_images[i] *  np.repeat(_labels[i,:,:,np.newaxis] > 0, 3,-1) for i in range(batch_size)]

			# visualize gradients of preds 
			# print(preds.shape)
			preds_w = preds.softmax(1).permute(0,2,3,1).detach().cpu().numpy()     # (B, H, W, C)
			preds_i = np.argmax(preds.detach().cpu().numpy(), 1).astype(np.uint8)  # (B, H, W)

			for i in range(batch_size):
				images_wandb.append(wandb.Image(_images[i], caption = "{}".format(image_path[i].split('/')[-1])))
				masks_wandb.append(wandb.Image( signals[i].astype(np.uint8), caption = "{}".format(image_path[i].split('/')[-1])))

				if len(images_wandb) >= max_uploads:
					break

		if config.debug and (step > 16):
			break

	if not os.path.exists(f"{dir2save}/plots"):
		os.makedirs(f"{dir2save}/plots", exist_ok=True)


	# confusion matrix including normal
	normalized_confusion_matrix = normalized_confusion_matrix.detach().cpu().numpy().transpose()
	normalized_confusion_matrix /= normalized_confusion_matrix.sum()

	# plot confusion matrix
	img_confusion_mat = plot_confusion_matrix(	normalized_confusion_matrix, 
												num_classes, 
												class_names, 
												f"{dir2save}/plots/{taskname}_confusion_matrix.png", 
												taskname)

	# confusion matrix excluding normal
	confusion_matrix = normalized_confusion_matrix[1:,1:]
	confusion_matrix /= confusion_matrix.sum()
	img_confusion_mat_excluding_normal = plot_confusion_matrix(	confusion_matrix, 
																num_classes-1, 
																class_names[1:], 
																f"{dir2save}/plots/{taskname}_confusion_matrix_excluding_normal.png", 
																taskname)				

	# # calculate iou and miou using normalized_confusion_matrix
	iou_all = []
	for i in range(num_classes):
		iou_all.append(metric['iou_sum_per_cls'][i] / metric['num_classes'][i])

	
	miou = metric['miou_sum'] / metric['num_images']
	miou_wo_bg = metric['miou_sum(w/o bg)'] / metric['num_images(w/o bg)']

	if wandb is not None:
		wandb.log({	f"{taskname}_miou_avg(w/ bg)": miou})	
		wandb.log({	f"{taskname}_miou_avg(no bg)": miou_wo_bg})
		for i in range(num_classes):
			wandb.log({	f"{taskname}_iou_{class_names[i]}_avg": iou_all[i]})
		wandb.log({	f"{taskname}_loss_avg": loss_sum/sample_num})


	if wandb is not None:
		wandb.log({	f"{taskname}_confusion_matrix": 				 wandb.Image(img_confusion_mat, caption = f"{taskname}_confusion_matrix"),
					f"{taskname}_confusion_matrix_excluding_normal": wandb.Image(img_confusion_mat_excluding_normal, caption = f"{taskname}_confusion_matrix_excluding_normal")
					})
		
	return loss_sum/sample_num, {'mIoU_all': miou, 'mIoU_wo_normal': miou_wo_bg, 'iou': iou_all}


def evaluate_ddp(	epoch, 
					config, 
					model, 
					loss_fn, 
					val_loader, 
					device, 
					threshold 		 = 0.5, 
					scheduler		 = None, 
					schd_loss_update = False, 
					wandb 			 = None,
					visualize 		 = False,
					max_uploads 	 = 64, 
					taskname 		 = 'test' # 'val' or 'test', 
					):
	classinfo = classInfo(include_normal = config.include_normal)

	dir2save = f"outputs/{config.expName}"

	model.eval()
	loss_sum 	= 0
	sample_num 	= 1e-10
	metric 		= None

	images_wandb = []
	masks_wandb  = []
	
	upload_image_wandb 	= False

	# class_names 	 = classID_inc_normal if config.include_normal else classID	
	class_names 	 = classinfo.get_class_names()
	num_classes 	 = classinfo.num_classes

	normalized_confusion_matrix 	= torch.zeros(num_classes, num_classes).to(device)
	normalized_confusion_matrix_i 	= torch.zeros(num_classes, num_classes).to(device)

	if config.rank in [-1, 0]:
		pbar = tqdm(enumerate(val_loader), total=len(val_loader))
	else:
		pbar = enumerate(val_loader)

	iou_all 	= []
	fname 		= ''
	n_images 	= 0
	miou 		= 0
	miou_sum 	= 0

	metric 		= { 'num_classes': [0]* num_classes, 
			 		'iou_sum_per_cls': [0]* num_classes,
			 		'miou_sum': 0,
					'num_images': 1e-6,
					'miou_sum(w/o bg)': 0,
					'num_images(w/o bg)': 1e-6
					}

	for step, (images, labels, image_path) in pbar:
		labels =  labels.to(device)
		preds   = model(images.float().to(device)).logits

		# calculate loss
		if loss_fn is not None:
			loss_cls = loss_fn['loss_cls'](preds, labels.long()).item()
			loss_iou = loss_fn['loss_iou'](F.softmax(preds, dim = 1).permute(0,2,3,1).float(),
											F.one_hot(labels.long(), num_classes).float()
										).item()
	
			loss = loss_cls + loss_iou
			loss_sum += loss*labels.shape[0]
		else:
			loss_iou = -1
			loss_cls = -1
			loss_sum = -1

		sample_num += labels.shape[0]

		# make confusion matrix for miou
		preds_i 	= torch.argmax(preds, 1)	# (B, C, H, W) to (B, H, W)
		labels_i 	= labels					# (B, H, W)

		for i, path in enumerate(image_path):
			_fname 		= path.split('/')[-1]
			pred_i 		= preds_i[i, :, :]
			label_i 	= labels_i[i, :, :]

			bincount 	= torch.bincount( pred_i.reshape(-1).long() * num_classes + label_i.long().reshape(-1),
										minlength = num_classes**2
										).reshape(num_classes, num_classes)
			
			if fname != _fname:
				if fname == '':
					fname = _fname
					continue

				# n_images 			+= 1
				metric['num_images'] += 1

				iou = []
				for j in range(num_classes):
					val = normalized_confusion_matrix_i[j, j] / (normalized_confusion_matrix_i[j, :].sum() + normalized_confusion_matrix_i[:, j].sum() - normalized_confusion_matrix_i[j, j])
					iou.append(val.item())
					metric['iou_sum_per_cls'][j] += val.item() if not np.isnan(val.item()) else 0
					metric['num_classes'][j] += 1 if not np.isnan(val.item()) else 0
				

				miou_each = np.nanmean(iou)
				if not np.isnan(np.nanmean(iou[1:])):
					metric['miou_sum(w/o bg)'] 		+= np.nanmean(iou[1:])
					metric['num_images(w/o bg)'] 	+= 1

				if not np.isnan(miou_each):
					metric['miou_sum'] += miou_each
				else:
					metric['num_images'] -= 1

				miou = metric['miou_sum'] / metric['num_images']	
				miou_no_bg = metric['miou_sum(w/o bg)'] / metric['num_images(w/o bg)'] if metric['num_images(w/o bg)'] > 0 else 0

				if config.rank in [-1, 0]:
					description = f'epoch {epoch}, {fname:35s}, loss({taskname}, cls|iou): {loss_cls:.4f}|{loss_iou:.4f}, miou(each): {miou_each:.4f}, miou(avg): {miou:.4f}, miou(avg,w/o bg): {miou_no_bg:.4f}'
					pbar.set_description(description)

				fname = _fname
				normalized_confusion_matrix_i[:, :] = bincount / (label_i.reshape(-1).shape[0]) 
				
			else:
				normalized_confusion_matrix_i[:, :] += bincount / (label_i.reshape(-1).shape[0]) 
				


		if config.rank in [-1, 0] and  visualize and (wandb is not None) and not config.debug:
			if (len(images_wandb) >= max_uploads) and (not upload_image_wandb):
				upload_image_wandb = True	
				wandb.log({	f"{taskname}_images": images_wandb,
							f"{taskname}_masks" : masks_wandb
							})			

							
		if visualize and (wandb is not None) and (len(images_wandb) < max_uploads):
			batch_size  	= images.shape[0]
			imgsize 		= labels.shape[1:3]

			_images = images.detach().cpu().numpy().reshape([batch_size, -1, images.shape[2],images.shape[3]])
			_images = np.transpose(_images, (0,2,3,1))
			_images = [undo_preprocess(_images[i,:,:,:], imgsize = imgsize)  for i in range(batch_size)]

			_labels  = labels.detach().cpu().numpy().reshape([batch_size, labels.shape[1], labels.shape[2]]).astype(np.uint8)

			# mask non-zero labels on _images
			signals = [	_images[i] *  np.repeat(_labels[i,:,:,np.newaxis] > 0, 3,-1) for i in range(batch_size)]

			# visualize gradients of preds 
			# print(preds.shape)
			preds_w = preds.softmax(1).permute(0,2,3,1).detach().cpu().numpy()     # (B, H, W, C)
			preds_i = np.argmax(preds.detach().cpu().numpy(), 1).astype(np.uint8)  # (B, H, W)

			for i in range(batch_size):
				images_wandb.append(wandb.Image(_images[i], caption = "{}".format(image_path[i].split('/')[-1])))
				masks_wandb.append(wandb.Image( signals[i].astype(np.uint8), caption = "{}".format(image_path[i].split('/')[-1])))

				if len(images_wandb) >= max_uploads:
					break

		if config.debug and (step > 16):
			break

	if not os.path.exists(f"{dir2save}/plots"):
		os.makedirs(f"{dir2save}/plots", exist_ok=True)


	# confusion matrix including normal
	normalized_confusion_matrix = normalized_confusion_matrix.detach().cpu().numpy().transpose()
	normalized_confusion_matrix /= normalized_confusion_matrix.sum()

	# plot confusion matrix
	img_confusion_mat = plot_confusion_matrix(	normalized_confusion_matrix, 
												num_classes, 
												class_names, 
												f"{dir2save}/plots/{taskname}_confusion_matrix_rank{config.rank}.png", 
												taskname)

	# confusion matrix excluding normal
	confusion_matrix = normalized_confusion_matrix[1:,1:]
	confusion_matrix /= confusion_matrix.sum()
	img_confusion_mat_excluding_normal = plot_confusion_matrix(	confusion_matrix, 
																num_classes-1, 
																class_names[1:], 
																f"{dir2save}/plots/{taskname}_confusion_matrix_excluding_normal_rank{config.rank}.png", 
																taskname)				

	# # calculate iou and miou using normalized_confusion_matrix
	iou_all = []
	for i in range(num_classes):
		if metric['num_classes'][i] > 0:
			iou_all.append(metric['iou_sum_per_cls'][i] / metric['num_classes'][i])
		else:
			iou_all.append(-1)
	
	miou = metric['miou_sum'] / metric['num_images']
	miou_wo_bg = metric['miou_sum(w/o bg)'] / metric['num_images(w/o bg)']

	if config.rank in [-1,0] and wandb is not None and not config.debug:
		wandb.log({	f"{taskname}_miou_avg(w/ bg)": miou})	
		wandb.log({	f"{taskname}_miou_avg(no bg)": miou_wo_bg})
		for i in range(num_classes):
			wandb.log({	f"{taskname}_iou_{class_names[i]}_avg": iou_all[i]})
		wandb.log({	f"{taskname}_loss_avg": loss_sum/sample_num})


	if config.rank in [-1,0] and wandb is not None and not config.debug:
		wandb.log({	f"{taskname}_confusion_matrix": 				 wandb.Image(img_confusion_mat, caption = f"{taskname}_confusion_matrix"),
					f"{taskname}_confusion_matrix_excluding_normal": wandb.Image(img_confusion_mat_excluding_normal, caption = f"{taskname}_confusion_matrix_excluding_normal")
					})
		
	return loss_sum/sample_num, {'mIoU_all': miou, 'mIoU_wo_normal': miou_wo_bg, 'iou': iou_all}

		