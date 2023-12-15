import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests

# feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-768-768")
# model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
# print(model.classifier)
# change number of classes to 11
# model.classifier = torch.nn.Conv2d(128, 11, kernel_size=(1, 1), stride=(1, 1))

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# # inputs = feature_extractor(images=image, return_tensors="pt")

# inputs = torch.rand(1, 3, 768, 768) 

# outputs = model(inputs)
# logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)

# print(model)


# print(inputs)

# print(type(model))
# save model
# torch.save(model, "segformer.pth")

# load model from pth
model = torch.load("segformer.pth")
print(model.decode_head.classifier)
# change number of classes to 10 
classes = 11
in_channels = model.decode_head.classifier.in_channels
model.decode_head.classifier = torch.nn.Conv2d(in_channels, classes, kernel_size=(1, 1), stride=(1, 1))
print(model.decode_head.classifier)

# model.classifier = torch.nn.Conv2d(128, 10, kernel_size=(1, 1), stride=(1, 1))

# change model to eval mode
# model.eval()
# print(model)

input_tensor = torch.rand(12, 3, 1024, 1024)
output = model(input_tensor).logits
print(output)
print(output.shape)
# print(output.logits.shape)

# print(type(model))