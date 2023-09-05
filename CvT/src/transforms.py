import numpy as np 
import cv2
import random
import torchlm

from albumentations import (
	HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
	Transpose, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
	IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
	IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, RandomCrop, 
	CoarseDropout, CenterCrop, Resize, ColorJitter, GaussianBlur,ElasticTransform, IAAAffine,
	KeypointParams, Rotate, PadIfNeeded, RandomShadow
)
from albumentations.pytorch import ToTensorV2

MEANPIXVAL = np.array([0.485, 0.456, 0.406])
STDPIXVAL  = np.array([0.229, 0.224, 0.225])

MEANPIXVAL_G = np.array([0.5])
STDPIXVAL_G  = np.array([0.25])



def get_valid_transforms(imgsize = 256, is_grayscale = False, is_tta=False):
		return Compose([
				RandomBrightnessContrast(brightness_limit=(-0.3,0.75), contrast_limit=(-0.5, 0.5), p=0.5) if is_tta else None,
				HorizontalFlip(p=0.5) if is_tta else None,
				Cutout(max_h_size = int(0.2*imgsize), max_w_size = int(0.2*imgsize), p=0.25) if is_tta else None,
				Resize(imgsize,imgsize),
				Normalize(mean = MEANPIXVAL_G if is_grayscale else MEANPIXVAL, 
						  std = STDPIXVAL_G if is_grayscale else STDPIXVAL_G, max_pixel_value = 255.0, p=1.0),				
				ToTensorV2()
			],  p = 1.,
				keypoint_params = KeypointParams(format='xy', remove_invisible=False)
			)

def get_train_transforms(imgsize = 256, is_grayscale = False):
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
				Resize(imgsize,imgsize),
				Normalize(mean = MEANPIXVAL_G if is_grayscale else MEANPIXVAL, 
						  std = STDPIXVAL_G if is_grayscale else STDPIXVAL_G, max_pixel_value = 255.0, p=1.0),
				# Normalize(mean = np.array([0.5,0.5,0.5]), std = np.array([0.5,0.5,0.5]), max_pixel_value=255.0, p=1.0)
				# CenterCrop(imgsize,imgsize)
				ToTensorV2()
			], 
			p = 1.,
			# keypoint_params = KeypointParams(format='xy', remove_invisible=False)
			)

	else:
		return Compose([
				PadIfNeeded(min_height=minsize, min_width=minsize),
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
						  std = STDPIXVAL_G if is_grayscale else STDPIXVAL_G, max_pixel_value = 255.0, p=1.0),
				ToTensorV2()
			], 
			p = 1.,
			# keypoint_params = KeypointParams(format='xy', remove_invisible=False)
			)




def pixel_jitter(src, p=0.5, max_=5.):
	# src             = src.astype(np.float16)
	pattern         = (np.random.rand(src.shape[0], src.shape[1],src.shape[2])-0.5)*2*max_
	img             = src + pattern
	img[img < 0]    = 0
	img[img > 255]  = 255
	return img

def Img_dropout(src,max_pattern_ratio=0.05):
	pattern         = np.ones_like(src)
	width_ratio     = random.uniform(0, max_pattern_ratio)
	height_ratio    = random.uniform(0, max_pattern_ratio)
	width           = src.shape[1]
	height          = src.shape[0]
	block_width     = width*width_ratio
	block_height    = height*height_ratio
	width_start     = int(random.uniform(0,width-block_width))
	width_end       = int(width_start+block_width)
	height_start    = int(random.uniform(0,height-block_height))
	height_end      = int(height_start+block_height)
	pattern[height_start:height_end,width_start:width_end,:] = 0
	img             = src*pattern
	return img


def adjust_contrast(image, factor):
	""" Adjust contrast of an image.
	Args
		image: Image to adjust.
		factor: A factor for adjusting contrast.
	"""
	mean = image.mean(axis=0).mean(axis=0)
	return _clip((image - mean) * factor + mean)


def adjust_brightness(image, delta):
	""" Adjust brightness of an image
	Args
		image: Image to adjust.
		delta: Brightness offset between -1 and 1 added to the pixel values.
	"""
	return _clip(image + delta * 255)


def adjust_hue(image, delta):
	""" Adjust hue of an image.
	Args
		image: Image to adjust.
		delta: An interval between -1 and 1 for the amount added to the hue channel.
			   The values are rotated if they exceed 180.
	"""
	image[..., 0] = np.mod(image[..., 0] + delta * 180, 180)
	return image


def adjust_saturation(image, factor):
	""" Adjust saturation of an image.
	Args
		image: Image to adjust.
		factor: An interval for the factor multiplying the saturation values of each pixel.
	"""
	image[..., 1] = np.clip(image[..., 1] * factor, 0, 255)
	return image


def _clip(image):
	"""
	Clip and convert an image to np.uint8.
	Args
		image: Image to clip.
	"""
	return np.clip(image, 0, 255).astype(np.uint8)

def _uniform(val_range):
	""" Uniformly sample from the given range.
	Args
		val_range: A pair of lower and upper bound.
	"""
	return np.random.uniform(val_range[0], val_range[1])


class ColorDistort():
	def __init__(
			self,
			contrast_range=(0.8, 1.2),
			brightness_range=(-.2, .2),
			hue_range=(-0.1, 0.1),
			saturation_range=(0.8, 1.2)
	):
		self.contrast_range = contrast_range
		self.brightness_range = brightness_range
		self.hue_range = hue_range
		self.saturation_range = saturation_range

	def __call__(self, image):
		if self.contrast_range is not None:
			contrast_factor = _uniform(self.contrast_range)
			image = adjust_contrast(image,contrast_factor)

		if self.brightness_range is not None:
			brightness_delta = _uniform(self.brightness_range)
			image = adjust_brightness(image, brightness_delta)

		if self.hue_range is not None or self.saturation_range is not None:

			image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

			if self.hue_range is not None:
				hue_delta = _uniform(self.hue_range)
				image = adjust_hue(image, hue_delta)

			if self.saturation_range is not None:
				saturation_factor = _uniform(self.saturation_range)
				image = adjust_saturation(image, saturation_factor)
			image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
		return image





SYMMETRY = [(0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9), (8, 8),
			(17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
			(31, 35), (32, 34),
			(36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
			(48, 54), (49, 53), (50, 52), (55, 59), (56, 58), (60, 64), (61, 63), (65, 67)]

def Mirror(src, label=None, symmetry=None):
	img = cv2.flip(src, 1)
	if label is None:
		return img, label

	width   = img.shape[1]
	cod     = []
	allc    = []
	for i in range(label.shape[0]):
		x, y = label[i][0], label[i][1]
		if x >= 0:
			x = width - 1 - x
		cod.append((x, y))

	# **** the joint index depends on the dataset ****
	for (q, w) in symmetry:
		cod[q], cod[w] = cod[w], cod[q]

	for i in range(label.shape[0]):
		allc.append(cod[i][0])
		allc.append(cod[i][1])
	label = np.array(allc).reshape(label.shape[0], 2)
	return img,label

######May wrong, when use it check it
def Rotate_aug(src, angle, label=None, center=None, scale=1.0):
	'''
	:param src: src image
	:param label: label should be numpy array with [[x1,y1],
													[x2,y2],
													[x3,y3]...]
	:param angle:
	:param center:
	:param scale:
	:return: the rotated image and the points
	'''
	image   = src
	(h, w)  = image.shape[:2]
	if center is None:
		center = (w / 2, h / 2)
	
	M = cv2.getRotationMatrix2D(center, angle, scale)
	if label is None:
		for i in range(image.shape[2]):
			image[:,:,i] = cv2.warpAffine(image[:,:,i], M, (w, h),
										  flags=cv2.INTER_CUBIC,
										  borderMode=cv2.BORDER_CONSTANT,
										  borderValue=[0,0,0])
		return image,None
	else:
		label=label.T
		####make it as a 3x3 RT matrix
		full_M=np.row_stack((M,np.asarray([0,0,1])))
		img_rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
									 borderMode=cv2.BORDER_CONSTANT, borderValue=[0,0,0])
		###make the label as 3xN matrix
		full_label = np.row_stack((label, np.ones(shape=(1,label.shape[1]))))
		label_rotated=np.dot(full_M,full_label)
		label_rotated=label_rotated[0:2,:]
		#label_rotated = label_rotated.astype(np.int32)
		label_rotated=label_rotated.T
		return img_rotated,label_rotated

def Affine_aug(src, strength, label=None):
	image       = src
	pts_base    = np.float32([[10,100],[200,50],[100,250]])
	pts1        = np.random.rand(3, 2) * random.uniform(-strength, strength) + pts_base
	pts1        = pts1.astype(np.float32)

	M           = cv2.getAffineTransform(pts1, pts_base)
	trans_img   = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]) ,
								borderMode=cv2.BORDER_CONSTANT,
								borderValue=[0,0,0])
	label_rotated = None

	if label is not None:
		label           = label.T
		full_label      = np.row_stack((label, np.ones(shape=(1, label.shape[1]))))
		label_rotated   = np.dot(M, full_label)
		label_rotated   = label_rotated.T

	return trans_img,label_rotated

