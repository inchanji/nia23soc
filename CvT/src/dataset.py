import sys 
import numpy as np 
import torch
from torch.utils.data import Dataset,DataLoader
import cv2
import itertools

from src.transforms import get_train_transforms, get_valid_transforms
from src.transforms import MEANPIXVAL, STDPIXVAL
from .FMix import sample_mask, make_low_freq_image, binarise_mask
from .augmix_utils import augment_and_mix

def load_data(path):
	img = cv2.imread(path) 
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	return img

def prepare_dataloader(df, config, is_training = True):
	dataset = ClassificationDataset(df,
								config,
								transforms 		= get_train_transforms(imgsize = config.imgsize, is_grayscale = config.is_grayscale) if is_training \
													else get_valid_transforms(imgsize = config.imgsize, is_grayscale = config.is_grayscale, is_tta = config.valid_tta), 
								is_train 	= is_training, 
								)

	data_loader 	= torch.utils.data.DataLoader(dataset, 
						batch_size 	= config.train_bs if is_training else config.valid_bs, 
						shuffle 	= True if is_training else False, 
						num_workers = config.num_workers)	


	return data_loader



class ClassificationDataset(Dataset):
	def __init__(self, 
				 df, 
				 config,
				 transforms 	= None,
				 output_label 	= True,
				 fmix_params 	= {
					 'alpha'		: 1.,
					 'decay_power'	: 3.,
					 'shape'		: (512, 512),
					 'max_soft'		: True,
					 'reformulate'	: False },
				 one_hot_label 	= True,
				 inference 		= False,
				 cutmix_params 	= { 'alpha': 1, },
				 is_train = True
				):

		super().__init__()
		self.df 		    = df.reset_index(drop=True).copy()
		self.file_names     = df['img_path'].values
		self.config 	    = config
		self.transforms     = transforms
		self.output_label   = output_label
		self.fmix_params    = fmix_params
		self.cutmix_params  = cutmix_params		
		self.one_hot_label  = one_hot_label
		self.inference      = inference		
		self.oof 		    = config.OOF
		self.is_train 		= is_train

		if self.config.do_fmix or self.config.do_cutmix:
			self.one_hot_label = True

		if self.config.do_fmix or self.config.do_cutmix:
			self.one_hot_label = True

		if self.inference:
			self.one_hot_label = False

		if self.output_label == True:
			self.labels = self.df['label'].values

			if self.one_hot_label is True and not self.inference:
				self.labels = np.eye(self.config.num_classes)[self.labels]

		if self.config.do_fmix_v2 or self.config.do_tile_mix:
			label_values = self.df['label'].values
			self.iter_labels = []
			for _ in range(self.config.num_classes):
				self.iter_labels.append([])

			for i, label in enumerate(label_values):
				self.iter_labels[label].append(i)

			for i in range(self.config.num_classes):
				np.random.shuffle(self.iter_labels[i])
				self.iter_labels[i] = itertools.cycle(self.iter_labels[i])


	def __len__(self):
		return self.df.shape[0]

	def __getitem__(self, index: int):
		# if not self.inference and CFG.maximize_rocauc:
		# 	while True:
		# 		w = self.weights[index]
		# 		if w > 0.8: 
		# 			index = np.random.choice(self.df.index, size=1)[0]
		# 		else:
		# 			break

		# get labels
		if self.output_label:
			target    = self.labels[index]

		# if self.oof:
		# 	loss_weight = self.loss_weights[index]

		file_path = self.file_names[index]

		img = load_data(file_path)

		#if self.aug_mix and not(self.inference):
		if self.config.aug_mix and np.random.uniform(0., 1., size=1)[0] > 0.5:
			img = augment_and_mix(img)

		if self.transforms:
			img = self.transforms(image=img.astype(np.float32))['image']
		else:
			img = img[np.newaxis,:,:]
			img = torch.from_numpy(img).float32()

		if not self.is_train and self.config.do_fmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
			with torch.no_grad():
				lam 		= np.clip(np.random.beta(self.fmix_params['alpha'], self.fmix_params['alpha']),0.6,0.7)
				# Make mask, get mean / std
				mask 		= make_low_freq_image(self.fmix_params['decay_power'], self.fmix_params['shape'])
				mask 		= binarise_mask(mask, lam, self.fmix_params['shape'], self.fmix_params['max_soft'])
				fmix_ix 	= np.random.choice(self.df.index, size=1)[0]
				fmix_img  	= load_data(self.file_names[fmix_ix])

				if self.transforms:
					fmix_img = self.transforms(image=fmix_img.astype(np.float32))['image']

				mask_torch = torch.from_numpy(mask)			
				# mix image
				img = mask_torch*img+(1.- mask_torch)*fmix_img
				# mix target
				rate   = mask.sum()/self.config.img_size/self.config.img_size
				target = rate * target + (1.-rate)*self.labels[fmix_ix]

		if not self.is_train and self.config.do_fmix_v2 and self.output_label and np.random.uniform(0., 1., size=1)[0] > 0.5:
			with torch.no_grad():
				for _ in range(1 if target == 0 else np.random.randint(1, 4)):
					lam = np.clip(np.random.beta(self.fmix_params['alpha'], self.fmix_params['alpha']),0.6,0.7)
					mask = make_low_freq_image(self.fmix_params['decay_power'], self.fmix_params['shape'])
					mask = binarise_mask(mask, lam, self.fmix_params['shape'], self.fmix_params['max_soft'])

					fmix_ix   = next(self.iter_labels[target])

					file_path = self.file_names[fmix_ix]
					fmix_img  = load_data(file_path)

					# if self.aug_mix and np.random.uniform(0., 1., size=1)[0] > 0.5:
					# 	fmix_img = augment_and_mix(fmix_img)

					if self.transforms:
						fmix_img = self.transforms(image=fmix_img.astype(np.float32))['image']

					mask_torch = torch.from_numpy(mask)
					img = mask_torch*img+(1.- mask_torch)*fmix_img
	

		if not self.is_train and self.config.do_cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
			#print(img.sum(), img.shape)
			with torch.no_grad():
				cmix_ix 	= np.random.choice(self.df.index, size=1)[0]
				cmix_img  	= load_data(self.file_names[cmix_ix])
				if self.transforms:
					cmix_img = self.transforms(image=cmix_img.astype(np.float32))['image']

				lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']),0.3,0.4)
				bbx1, bby1, bbx2, bby2 = rand_bbox((self.config.img_size, self.config.img_size), lam)

				img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

				rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (self.config.img_size * self.config.img_size))
				target = rate*target + (1.-rate)*self.labels[cmix_ix]


		if not self.is_train and self.config.do_tile_mix and np.random.uniform(0., 1., size=1)[0] > 0.5:
			with torch.no_grad():
				n_tiles 		= np.random.choice([i+1 for i in range(self.config.tile_mix_depth)])
				tile_size_x 	= int(self.config.img_size/n_tiles)
				tile_size_y 	= int(self.config.img_size/n_tiles)

				for i_mix in range(n_tiles*n_tiles):
					if np.random.uniform(0., 1., size=1)[0] > 0.5: continue
					mix_ix 		= next(self.iter_labels[target])

					file_path 	= self.file_names[mix_ix]
					mix_img  	= load_data(file_path)
					x_i 		= int((i_mix // n_tiles) * tile_size_x + 0.5 * tile_size_x)
					y_i 		= int((i_mix % n_tiles) * tile_size_y + 0.5 * tile_size_y)

					lam 		= np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']),0.3,0.4)	
					bbx1, bby1, bbx2, bby2 = rand_bbox_v2((tile_size_x, tile_size_y), lam, (x_i,y_i), (self.config.img_size, self.config.img_size))

					if self.config.aug_mix and np.random.uniform(0., 1., size=1)[0] > 0.5:
						mix_img = augment_and_mix(mix_img)

					if self.transforms:
						mix_img = self.transforms(image=mix_img.astype(np.float32))['image']						

					if np.random.uniform(0., 1., size=1)[0] > 0.25:
						img[:, bbx1:bbx2, bby1:bby2] = mix_img[:, bbx1:bbx2, bby1:bby2]
					else:
						lam = np.clip(np.random.beta(self.fmix_params['alpha'], self.fmix_params['alpha']),0.6,0.7)
						mask = make_low_freq_image(self.fmix_params['decay_power'], self.fmix_params['shape'])
						mask = binarise_mask(mask, lam, self.fmix_params['shape'], self.fmix_params['max_soft'])
						mask_torch = torch.from_numpy(mask)			
						mix_img = mask_torch*img+(1.- mask_torch)*mix_img
						img[:, bbx1:bbx2, bby1:bby2] = mix_img[:, bbx1:bbx2, bby1:bby2]

		if self.output_label == True:
			return img.float(), target
		else:
			return img
