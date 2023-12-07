import os
import cv2
import glob 
import torch
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.transforms import get_train_transforms, get_valid_transforms
import torch.nn.functional as F

colors_dict = {	
	0 : [0,255,0],   		# bg
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

classID 		   = ['crack','reticular crack','detachment','spalling','efflorescence','leak','rebar','material separation','exhilaration','damage' ]
classID_inc_normal = ['normal','crack','reticular crack','detachment','spalling','efflorescence','leak','rebar','material separation','exhilaration','damage']

class classInfo():
	def __init__(self, num_classes = len(classID), include_normal = False):
		self.num_classes 	= num_classes
		self.classID 		= classID
		self.class2idx 		= {classID[i]:i for i in range(num_classes)}
		self.idx2class 		= {i:classID[i] for i in range(num_classes)}
		self.typoclass 		= {"efflorescene": "efflorescence"}

		if include_normal:
			self.include_normal()
	
		self.inculde_typo()	
		

	def inculde_typo(self):
		typoclass = {}
		for key, value in self.typoclass.items():
			self.class2idx.update({key:self.class2idx[value]})
	
	def include_normal(self):
		self.num_classes += 1
		self.classID = ['normal'] + self.classID
		self.class2idx = {self.classID[i]:i for i in range(self.num_classes)}
		self.idx2class = {i:self.classID[i] for i in range(self.num_classes)}

	def get_class_names(self):
		return self.classID
	

	def get_crack_idx(self):
		return [self.class2idx['crack'], self.class2idx['reticular crack']]
	
	def __len__(self):
		return self.num_classes


def prepare_dataloader_ddp(df, config, is_training = True):
	# distributed data sampler
	dataset = SegmentationDataset(df,
									config,
									transforms 		= get_train_transforms(imgsize = config.imgsize, is_grayscale = config.is_grayscale) if is_training \
												 else get_valid_transforms(imgsize = config.imgsize, is_grayscale = config.is_grayscale, is_tta = config.valid_tta),
									is_train 	= is_training, 
								)
	
	data_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas = config.world_size, rank = config.rank)
	data_loader  = torch.utils.data.DataLoader(dataset,
						batch_size 	= config.train_bs if is_training else config.valid_bs, 
						shuffle 	= False, 
						num_workers = config.num_workers,
						pin_memory 	= True,
						sampler 	= data_sampler)
	
	return data_loader
	

def prepare_dataloader(df, config, is_training = True):
	dataset = SegmentationDataset(df,
									config,
									transforms 		= get_train_transforms(imgsize = config.imgsize, is_grayscale = config.is_grayscale) if is_training \
												 else get_valid_transforms(imgsize = config.imgsize, is_grayscale = config.is_grayscale, is_tta = config.valid_tta),
									is_train 	= is_training, 
								)

	data_loader 	= 	torch.utils.data.DataLoader(dataset, 
							batch_size 	= config.train_bs if is_training else config.valid_bs, 
							shuffle 	= True if is_training else False, 
							num_workers = config.num_workers
						)	
	return data_loader

def construct_patch_df(df0, patchsize = 512, reticular_crack = 2, vis = True):
	df 			= pd.DataFrame()
	x0 			= []
	y0 			= []
	width 		= []
	height 		= []
	img_path 	= []
	label 		= []
	class_ 		= []

	if type(patchsize) == tuple:
		patchsize = patchsize[0]

	print("construct patch dataframe")
	if vis:
		pbar = tqdm(range(len(df0)))
	else:
		pbar = range(len(df0))
	for i in pbar:
		row, col = df0.iloc[i]['img_h'], df0.iloc[i]['img_w']
		cls_ = df0.iloc[i]['class']

		if type(cls_) == str:
			cls_ = cls_.split(',')
			cls_ = [int(cls) for cls in cls_]
		else:
			cls_ = [cls_]

		# print(cls_	)
		# check if cls_ contains 1 or 2
		if reticular_crack in cls_:
			x0.append(0)
			y0.append(0)	
			width.append(col)	
			height.append(row)	
			img_path.append(df0.iloc[i]['img_path'])
			label.append(df0.iloc[i]['label'])
			class_.append(df0.iloc[i]['class'])			
		else:	
			# get index of row and col using patchsize and row, col
			# print(row, col, patchsize)
			split_row = int(row) // patchsize
			split_col = int(col) // patchsize

			for r in range(split_row):
				for c in range(split_col):
					x0.append(max(int(np.floor(col/split_col * c)),0))
					y0.append(max(int(np.floor(row/split_row * r)),0))
					width.append(min(int(np.ceil(col/split_col * (c+1))), col))
					height.append(min(int(np.ceil(row/split_row * (r+1))), row))
					img_path.append(df0.iloc[i]['img_path'])
					label.append(df0.iloc[i]['label'])
					class_.append(df0.iloc[i]['class'])

	df['img_path'] 	= img_path
	df['label'] 	= label
	df['class'] 	= class_	

	df['x0'] 		= x0
	df['y0'] 		= y0
	df['width'] 	= width
	df['height'] 	= height
	
	return df


# dataset class for segmentation dataset
class SegmentationDataset(Dataset):
	def __init__(self, df, config, transforms = None, is_train = True, sparse_aug = True, reduce_factor = 4):
		self.df 			 = df
		self.config 		 = config
		self.transforms 	 = transforms
		self.is_train 		 = is_train
		self.sparse_aug 	 = sparse_aug	
		self.imgsize 		 = config.imgsize
		self.enable_patch 	 = config.enable_patch
		self.reticular_crack = -1
		self.reduce_factor  = reduce_factor

		if is_train:
			self.df 			= df
		elif self.enable_patch:
			self.df 			= construct_patch_df(df, patchsize = self.imgsize, reticular_crack = self.reticular_crack, vis = True if config.rank in [-1, 0] else False)
		else:
			self.df 			= df
		

	def __getitem__(self, idx):
		image_path 	= self.df.iloc[idx]['img_path']
		mask_path  	= self.df.iloc[idx]['label']
		class_ 		= self.df.iloc[idx]['class']

		image 		= cv2.imread(image_path)
		mask  		= cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


		if not self.is_train and self.enable_patch:
			x0, y0, width, height = self.df.iloc[idx][['x0', 'y0', 'width', 'height']].values
			image = image[y0:y0+height, x0:x0+width]
			mask  = mask[y0:y0+height, x0:x0+width]

		if self.is_train and self.enable_patch:
			if type(class_) == str:
				cls = class_.split(',')
				cls = [int(c) for c in cls]
			else:
				cls = [class_]
			
			if self.reticular_crack in cls:
				image = image
				mask  = mask
			else:
				xc, yc = self.df.iloc[idx][['x_cen', 'y_cen']].values
				imgsize_h = int(0.5 * self.imgsize * (1 + random.uniform(-0.5, 0.5)))
				imgsize_w = int(0.5 * self.imgsize * (1 + random.uniform(-0.5, 0.5)))

				# check nan values
				if np.isnan(xc) or np.isnan(yc):
					xc = image.shape[1] // 2
					yc = image.shape[0] // 2

				
				x0 = int(xc - imgsize_w); y0 = int(yc - imgsize_h)
				x1 = int(xc + imgsize_w); y1 = int(yc + imgsize_h)
				x0 = max(x0, 0); y0 = max(y0, 0)
				x1 = min(x1, image.shape[1]); y1 = min(y1, image.shape[0])

				image = image[y0:y1, x0:x1]
				mask  = mask[y0:y1, x0:x1]
		



		# if self.transforms:
		sample 	= {'image': image, 'mask': mask}
		sample 	= self.transforms(**sample)
		image, mask = sample['image'], sample['mask']
		# else:
		# 	image = cv2.resize(image, (self.config.imgsize, self.config.imgsize))
		# 	mask  = cv2.resize(mask, (self.config.imgsize, self.config.imgsize))

		# round mask value after resize operation
		mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size = (self.config.imgsize // self.reduce_factor, self.config.imgsize // self.reduce_factor), mode = 'nearest').squeeze(0).squeeze(0)
		mask = mask.round()
	
		if self.is_train:
			return image, mask, image_path
		else:
			return image, mask, image_path

	def __len__(self):
		return len(self.df)
	

	
	



class MixUpPaste:
	def __init__(self, 
	      		root_path,
		 		p: float 			 = 1.0,
				max_num:int 		 = 3,
				scale: list 		 = [0.5, 1.5], 
				iou_threshold: float = 0.1,
				max_try: int 		 = 10
				) -> None:
		
		if root_path is None:
			self.imgs 	= []
			classes 	= []
		else:
			self.imgs = glob(f"{root_path}/*")
			classes = [os.path.basename(img).split(".")[0].split("_")[0] for img in self.imgs]

		# get unique classes
		self.classes = list(set(classes))

		# get counts for each class
		self.counts = {c: classes.count(c) for c in self.classes}

		# least common multiple of counts
		self.lcm = np.lcm.reduce(list(self.counts.values()))

		# repeat each class to make the counts the same
		for c in self.classes:
			self.imgs += [img for img in self.imgs if os.path.basename(img).split(".")[0].split("_")[0] == c] * (self.lcm // self.counts[c])

		# shuffle
		random.shuffle(self.imgs)

		self.idx = 0
		self.p = p
		self.scale = scale # scale of the bbox
		self.max_num = max_num # max number of bbox to be pasted
		self.iou_threshold = iou_threshold
		self.max_try = max_try

	def __call__(self, labels: dict) -> dict:
		img 		= labels["image"]
		h, w, c 	= img.shape
		bboxes 		= labels["bboxes"]
		category_id = labels["category_id"]	

		num_pasted_noise = 0
		num_try = 0

		if random.random() < self.p:
			while num_pasted_noise < self.max_num and num_try < self.max_try:
				noise_img = cv2.imread(self.imgs[self.idx])
				self.idx += 1
				self.idx %= len(self.imgs)	

				noise_img = cv2.cvtColor(noise_img, cv2.COLOR_BGR2RGB)
				noise_h, noise_w, noise_c = noise_img.shape

				# random scale
				scale = random.uniform(self.scale[0], self.scale[1])
				noise_img = cv2.resize(noise_img, (int(noise_w*scale), int(noise_h*scale)))
				noise_h, noise_w, noise_c = noise_img.shape

				# random position
				x1 = random.randint(0, w-1-noise_w)
				y1 = random.randint(0, h-1-noise_h)
				x2 = x1 + noise_w
				y2 = y1 + noise_h

				# check if the bbox is overlapped with other bbox with iou > 0.5
				# if overlapped, skip
				overlapped = False
				for bbox in bboxes:
					iou = self.compute_iou([x1, y1, x2, y2], bbox)
					if iou > self.iou_threshold:
						overlapped = True
						break

				# paste noise
				if not overlapped:
					img[y1:y2, x1:x2] = noise_img
					num_pasted_noise += 1
				num_try += 1

		labels["image"] = img
		return labels
	
	def compute_iou(self, bbox1: list, bbox2: list) -> float:
		"""
		Compute the intersection over union of two set of bboxes wrt bbox1, 
		each bbox is represented as [x1, y1, x2, y2]
		"""
		x1 = max(bbox1[0], bbox2[0])
		y1 = max(bbox1[1], bbox2[1])
		x2 = min(bbox1[2], bbox2[2])
		y2 = min(bbox1[3], bbox2[3])

		intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
		area1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
		area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
		return intersection / area1
	
	def __len__(self):
		return len(self.imgs)