from torchvision.transforms import ColorJitter
from transformers import SegformerFeatureExtractor, SegformerImageProcessor
from transformers import SegformerForSemanticSegmentation
from transformers import TrainingArguments
from transformers import Trainer
import torch
from torch import nn
import evaluate
import numpy as np
from .config import *



feature_extractor = SegformerImageProcessor(do_reduce_labels=True)
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1) 

def train_transforms(example_batch):
	images = [jitter(x) for x in example_batch['pixel_values']]
	labels = [x for x in example_batch['label']]
	inputs = feature_extractor(images, labels)
	return inputs


def val_transforms(example_batch):
	images = [x for x in example_batch['pixel_values']]
	labels = [x for x in example_batch['label']]
	inputs = feature_extractor(images, labels)
	return inputs


# Refer to https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/trainer#transformers.TrainingArguments 
# for more details on the Trainer arguments
training_args = TrainingArguments(
	"segformer-b0-finetuned-segments-sidewalk-outputs",
	learning_rate=lr,
	num_train_epochs=epochs,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size*2,
	dataloader_num_workers=6,
	save_total_limit=3,
	evaluation_strategy="steps",
	save_strategy="steps",
	save_steps=20,
	eval_steps=20,
	logging_steps=1,
	eval_accumulation_steps=2,
	load_best_model_at_end=True,
	push_to_hub=False,
	resume_from_checkpoint=True,
	metric_for_best_model="eval_val_loss",
	#hub_model_id=hub_model_id,
	#hub_strategy="end",
)



# def compute_metrics(_dataset):
# 	metric = evaluate.load("mean_iou")
# 	metrics = {}

# 	eval_sets = ['val', 'test']

# 	with torch.no_grad():
		
# 		for eval_set in eval_sets:
# 			# valid set 
# 			logits, labels = _dataset[eval_set]

# 			logits_tensor 	= torch.from_numpy(logits)
# 			total_size 		= len(logits_tensor)

# 			# calculate metrics by batch
# 			num_batches 	= total_size // batch_size

# 			mean_iou 		 = None
# 			per_category_iou = None

# 			valid_batches_per_cls = np.zeros(len(id2label))
# 			valid_batches_tot 	  = 0

# 			for i in range(num_batches):
# 				start = i * batch_size
# 				end = (i+1) * batch_size
# 				if end > total_size:
# 					end = total_size

# 				logits_batch = logits_tensor[start:end]
# 				labels_batch = labels[start:end]
# 				iter_size 	 = end-start
				
# 				logits_batch = nn.functional.interpolate(
# 						logits_batch,
# 						size=labels_batch.shape[-2:],
# 						mode="bilinear",
# 						align_corners=False,
# 					).argmax(dim=1)
# 				pred_labels_batch = logits_batch.detach().cpu().numpy()

# 				# see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
# 				metrics_batch = metric._compute(
# 							predictions=pred_labels_batch,
# 							references=labels_batch,
# 							num_labels=len(id2label),
# 							ignore_index=0,
# 							reduce_labels=feature_extractor.do_reduce_labels,
# 							)
				
# 				if np.isfinite(metrics_batch["mean_iou"]):
# 					valid_batches_tot += iter_size

# 				metrics_batch["mean_iou"] = np.nan_to_num(metrics_batch["mean_iou"])

# 				if mean_iou is None:
# 					mean_iou = metrics_batch["mean_iou"] * iter_size
# 				else:
# 					mean_iou += metrics_batch["mean_iou"] * iter_size

# 				# replace nan with 0 in metrics_batch["per_category_iou"]
# 				valid_batches_per_cls += np.isfinite(metrics_batch["per_category_iou"]).astype(np.int32) * iter_size

# 				metrics_batch["per_category_iou"] = np.nan_to_num(metrics_batch["per_category_iou"])
				
# 				if per_category_iou is None:
# 					per_category_iou = metrics_batch["per_category_iou"] * iter_size
# 				else:
# 					per_category_iou += metrics_batch["per_category_iou"] * iter_size

# 			per_category_iou = per_category_iou / valid_batches_per_cls
# 			mean_iou = mean_iou / valid_batches_tot

			
# 			metrics[f"mean_iou({eval_set})"] = mean_iou
# 			for i, v in enumerate(per_category_iou):
# 				metrics[f"iou_{id2label[i+1]}({eval_set})"] = v
	
# 	return metrics



def compute_metrics(eval_pred):
	metric = evaluate.load("mean_iou")
	with torch.no_grad():
		logits, labels = eval_pred
		# print(">>> LABELS", np.unique(labels))
		logits_tensor = torch.from_numpy(logits)
		# print(">>> LOGITS shape:", logits_tensor.shape)

		total_size = len(logits_tensor)
		# calculate metrics by batch
		num_batches = total_size // batch_size
		#num_batches = 1
		

		mean_iou = None
		# mean_accuracy = None
		# overall_accuracy = None
		per_category_iou = None
		# per_category_accuracy = None

		valid_batches_per_cls = np.zeros(len(id2label))
		valid_batches_tot = 0

		for i in range(num_batches):
			start = i * batch_size
			end = (i+1) * batch_size
			if end > total_size:
				end = total_size
			logits_batch = logits_tensor[start:end]
			labels_batch = labels[start:end]
			iter_size = end-start
			
			logits_batch = nn.functional.interpolate(
					logits_batch,
					size=labels_batch.shape[-2:],
					mode="bilinear",
					align_corners=False,
				).argmax(dim=1)
			pred_labels_batch = logits_batch.detach().cpu().numpy()

			# currently using _compute instead of compute
			# see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
			metrics_batch = metric._compute(
						predictions=pred_labels_batch,
						references=labels_batch,
						num_labels=len(id2label),
						ignore_index=0,
						reduce_labels=feature_extractor.do_reduce_labels,
						)
			
			if np.isfinite(metrics_batch["mean_iou"]):
				valid_batches_tot += iter_size

			metrics_batch["mean_iou"] = np.nan_to_num(metrics_batch["mean_iou"])

			if mean_iou is None:
				mean_iou = metrics_batch["mean_iou"] * iter_size
			else:
				mean_iou += metrics_batch["mean_iou"] * iter_size

			# replace nan with 0 in metrics_batch["per_category_iou"]
			valid_batches_per_cls += np.isfinite(metrics_batch["per_category_iou"]).astype(np.int32) * iter_size

			metrics_batch["per_category_iou"] = np.nan_to_num(metrics_batch["per_category_iou"])
			
			if per_category_iou is None:
				per_category_iou = metrics_batch["per_category_iou"] * iter_size
			else:
				per_category_iou += metrics_batch["per_category_iou"] * iter_size

		per_category_iou = per_category_iou / valid_batches_per_cls
		mean_iou = mean_iou / valid_batches_tot


		metrics = {}
		metrics["mean_iou"] = mean_iou
		for i, v in enumerate(per_category_iou):
			metrics[f"iou_{id2label[i+1]}"] = v
	
	return metrics