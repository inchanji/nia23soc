import numpy as np 
import cv2
import random

from albumentations.pytorch import ToTensorV2

from albumentations import (
	HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
	Transpose, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
	IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
	IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, RandomCrop, 
	CoarseDropout, CenterCrop, Resize, ColorJitter, GaussianBlur,ElasticTransform, IAAAffine,
	KeypointParams, Rotate, PadIfNeeded, RandomShadow
)



MEANPIXVAL = np.array([0.485, 0.456, 0.406])
STDPIXVAL  = np.array([0.229, 0.224, 0.225])

MEANPIXVAL_G = np.array([0.5])
STDPIXVAL_G  = np.array([0.25])


def get_valid_transforms(imgsize = 512, is_grayscale = False, is_tta=False):
		return Compose([
				# RandomBrightnessContrast(brightness_limit=(-0.3,0.75), contrast_limit=(-0.5, 0.5), p=0.5) if is_tta else None,
				# HorizontalFlip(p=0.5) if is_tta else None,
				# Cutout(max_h_size = int(0.2*imgsize), max_w_size = int(0.2*imgsize), p=0.25) if is_tta else None,
				Resize(imgsize,imgsize),
				Normalize(mean = MEANPIXVAL_G if is_grayscale else MEANPIXVAL, 
						  std = STDPIXVAL_G if is_grayscale else STDPIXVAL, max_pixel_value = 255.0, p=1.0),				
				ToTensorV2()
			],  p = 1.)

def get_train_transforms(imgsize = 512, is_grayscale = False):
	minsize = int(imgsize * 1.05)
	if is_grayscale:
		return Compose([
				# PadIfNeeded(min_height=minsize, min_width=minsize),
				RandomBrightnessContrast(brightness_limit=(-0.3,0.75), contrast_limit=(-0.5, 0.5), p=0.5),
				OneOf([ GaussianBlur(p=0.5),
						GaussNoise(p=0.5),
						Cutout(max_h_size = int(0.2*imgsize), max_w_size = int(0.2*imgsize), p=0.25)]),
				# VerticalFlip(p=0.5),
				HorizontalFlip(p=0.5), 
				OneOf([
					ShiftScaleRotate(rotate_limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT),
					# Rotate(limit=15, p=0.5),
					IAAAffine(shear=5, rotate=45.0, p=0.5, mode='constant')
					]),
				# OneOf([
				# 	# CLAHE(p=0.5),
				# 	Resize(imgsize,imgsize),
				# 	RandomResizedCrop(imgsize, imgsize, always_apply = True, p=1),
				# 	]),

				Resize(imgsize, imgsize),
				
				Normalize(mean = MEANPIXVAL_G if is_grayscale else MEANPIXVAL, 
						  std = STDPIXVAL_G if is_grayscale else STDPIXVAL, max_pixel_value = 255.0, p=1.0),
				# Normalize(mean = np.array([0.5,0.5,0.5]), std = np.array([0.5,0.5,0.5]), max_pixel_value=255.0, p=1.0)
				# CenterCrop(imgsize,imgsize)
				ToTensorV2()
			], 
			p = 1.,
			)

	else:
		return Compose([
				OneOf([
					HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.1),
					ColorJitter(p=0.1),
					RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
					]),
				
				OneOf([
					# CLAHE(p=0.5),
					Compose([GaussianBlur(p=0.5),GaussNoise(p=0.5),Cutout(p=0.1)],p = 1.),
					]),

				OneOf([
					HorizontalFlip(p=0.5),
					# Rotate(limit=30, p=0.5),
					ShiftScaleRotate(rotate_limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT),
					IAAAffine(shear=15, p=0.5, mode='constant')
					]),
				Resize(imgsize,imgsize),
				Normalize(mean = MEANPIXVAL_G if is_grayscale else MEANPIXVAL, 
						  std = STDPIXVAL_G if is_grayscale else STDPIXVAL, max_pixel_value = 255.0, p=1.0),
				ToTensorV2()
			], 
			p = 1.,
			
			)