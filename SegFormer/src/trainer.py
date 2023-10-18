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
	#hub_model_id=hub_model_id,
	#hub_strategy="end",
)

items = ['mean_iou', 'per_category_iou', 'per_category_accuracy']

metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
	with torch.no_grad():
		logits, labels = eval_pred
		print(">>> LABELS", np.unique(labels))
		logits_tensor = torch.from_numpy(logits)
		print(">>> LOGITS shape:", logits_tensor.shape)

		total_size = len(logits_tensor)
		# calculate metrics by batch
		num_batches = total_size // batch_size
		#num_batches = 1
		

		mean_iou = None
		mean_accuracy = None
		overall_accuracy = None
		per_category_iou = None
		per_category_accuracy = None


		for i in range(num_batches):
			start = i * batch_size
			end = (i+1) * batch_size
			if end > total_size:
				end = total_size
			logits_batch = logits_tensor[start:end]
			labels_batch = labels[start:end]
			
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

			# print("METRICS", metrics_batch)

			if mean_iou is None:
				mean_iou = metrics_batch["mean_iou"] * (end-start)
			else:
				mean_iou += metrics_batch["mean_iou"] * (end-start)	

			if per_category_iou is None:
				per_category_iou = metrics_batch["per_category_iou"] * (end-start)
			else:
				per_category_iou += metrics_batch["per_category_iou"] * (end-start)

		
		# # scale the logits to the size of the label
		# logits_tensor = nn.functional.interpolate(
        #             logits_tensor,
        #             size=labels.shape[-2:],
        #             mode="bilinear",
        #             align_corners=False,
        #         ).argmax(dim=1)

		# pred_labels = logits_tensor.detach().cpu().numpy()
		# print("PRED LABELS", np.unique(pred_labels))
		
		# # currently using _compute instead of compute
		# # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
		# metrics = metric._compute(
		# 			predictions=pred_labels,
		# 			references=labels,
		# 			num_labels=len(id2label),
		# 			ignore_index=0,
		# 			reduce_labels=feature_extractor.do_reduce_labels,
				# )
		
		# print("METRICS", metrics)
		
		per_category_iou = per_category_iou / total_size
		mean_iou = mean_iou / total_size


		# add per category metrics as individual key-value pairs
		# per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
		# per_category_iou = metrics.pop("per_category_iou").tolist()

		metrics = {}
		metrics["mean_iou"] = mean_iou
		for i, v in enumerate(per_category_iou):
			metrics[f"iou_{id2label[i+1]}"] = v
		# metrics.update({f"accuracy_{id2label[i+1]}": v for i, v in enumerate(per_category_accuracy)})
		# metrics.update({f"iou_{id2label[i+1]}": v for i, v in enumerate(per_category_iou)})
	
	return metrics