import time
import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from src.transforms import MEANPIXVAL, STDPIXVAL
import torch.nn.functional as F
import cv2
from src.dataset import classID, classID_inc_normal
# from matplotlib import pyplot as plt
from src.plot_utils import plot_confusion_matrix


colors_dict = {
	
	0 : [0,0,0],   		# bg
	1 : [220,20,60],
	2 : [119,11,32],
	3 : [0,0,142],
	4 : [0,0,230],
	5 : [0,60,100],
	6 : [0,0,230],
	7 : [0,80,100],
	8 : [0,0,70],
	9 : [0,0,230],
	10 : [250,170,30]
}

# classeID = {
#     'crack': 0,                 # 균열
#     'reticular crack': 1,       # 망상균열
#     'detachment': 2,            # 박리
#     'spalling': 3,              # 박락
#     'efflorescene': 4,          # 벡태(typo)
#     'efflorescence': 4,         # 벡태
#     'leak': 5,                  # 누수
#     'rebar': 6,                 # 철근노출
#     'material separation': 7,   # 재료분리
#     'exhilaration': 8,          # 들뜸
#     'damage': 9,                # 파손
# }

    
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
	# switch to train mode
	model.train()
	if config.apex:
		scaler = GradScaler()	
	
	running_loss = None
	sample_num 		= 0
	loss_sum 		= 0

	images_wandb 	= []
	masks_wandb  	= []	

	num_classes 	 = config.num_classes + 1 if config.include_normal else config.num_classes
	class_names 	 = classID_inc_normal if config.include_normal else classID
	
	# pytorch version of confusion matrix
	normalized_confusion_matrix = torch.zeros(num_classes, num_classes).to(device)

	pbar = tqdm(enumerate(train_loader), total=len(train_loader))

	for step, (images, labels, image_path) in pbar:
		# if step > 16: 
		# 	break

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
		loss_iou = loss_fn['loss_iou'](F.softmax(preds, dim = 1).float(), 
							  		F.one_hot(labels.long(), num_classes).permute(0,3,1,2).float()
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



	# confusion matrix including normal
	normalized_confusion_matrix = normalized_confusion_matrix.detach().cpu().numpy().transpose()
	normalized_confusion_matrix /= normalized_confusion_matrix.sum()

	# plot confusion matrix
	img_confusion_mat = plot_confusion_matrix(normalized_confusion_matrix, num_classes, class_names, f"plots/{taskname}_confusion_matrix.png", taskname)

	# confusion matrix excluding normal
	confusion_matrix = normalized_confusion_matrix[1:,1:]
	confusion_matrix /= confusion_matrix.sum()
	img_confusion_mat_excluding_normal = plot_confusion_matrix(confusion_matrix, num_classes-1, class_names[1:], f"plots/{taskname}_confusion_matrix_excluding_normal.png", taskname)

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
	model.eval()
	loss_sum 	= 0
	sample_num 	= 1e-10
	metric 		= None

	images_wandb = []
	masks_wandb  = []
	
	upload_image_wandb 	= False

	num_classes 	 = config.num_classes + 1 if config.include_normal else config.num_classes
	class_names 	 = classID_inc_normal if config.include_normal else classID	

	normalized_confusion_matrix = torch.zeros(num_classes, num_classes).to(device)

	pbar = tqdm(enumerate(val_loader), total=len(val_loader))

	for step, (images, labels, image_path) in pbar:
		# if step > 16: 
		# 	break		
		labels =  labels.to(device)
		preds   = model(images.float().to(device)).logits

		# calculate loss
		loss_cls = loss_fn['loss_cls'](preds, labels.long())
		loss_iou = loss_fn['loss_iou'](F.softmax(preds, dim = 1).float(), 
										F.one_hot(labels.long(), num_classes).permute(0,3,1,2).float()
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

		if ((step + 1) %  config.accum_iter == 0) or ((step + 1) == len(val_loader)):
			description = f'epoch {epoch} loss({taskname}, cls/iou): {loss_cls.item():.4f}, {loss_iou.item():.4f}'
			pbar.set_description(description)

								

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
		
		if visualize and (wandb is not None):
			if (len(images_wandb) >= max_uploads) and (not upload_image_wandb):
				upload_image_wandb = True	
				wandb.log({	f"{taskname}_images": images_wandb,
							f"{taskname}_masks" : masks_wandb
							})
				
	# confusion matrix including normal
	normalized_confusion_matrix = normalized_confusion_matrix.detach().cpu().numpy().transpose()
	normalized_confusion_matrix /= normalized_confusion_matrix.sum()

	# plot confusion matrix
	img_confusion_mat = plot_confusion_matrix(normalized_confusion_matrix, num_classes, class_names, f"plots/{taskname}_confusion_matrix.png", taskname)

	# confusion matrix excluding normal
	confusion_matrix = normalized_confusion_matrix[1:,1:]
	confusion_matrix /= confusion_matrix.sum()
	img_confusion_mat_excluding_normal = plot_confusion_matrix(confusion_matrix, num_classes-1, class_names[1:], f"plots/{taskname}_confusion_matrix_excluding_normal.png", taskname)				

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


	if wandb is not None:
		wandb.log({	f"{taskname}_confusion_matrix": 				 wandb.Image(img_confusion_mat, caption = f"{taskname}_confusion_matrix"),
					f"{taskname}_confusion_matrix_excluding_normal": wandb.Image(img_confusion_mat_excluding_normal, caption = f"{taskname}_confusion_matrix_excluding_normal")
					})
		
	return loss_sum/sample_num, {'mIoU_all': miou_all, 'mIoU_wo_normal': miou_wo_normal, 'iou': iou_all}


