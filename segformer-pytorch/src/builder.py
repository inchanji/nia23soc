import torch
from transformers import SegformerForSemanticSegmentation




def load_model(model_path, num_classes=11, device="cuda"):
    """Build model."""
    model = torch.load(model_path, map_location=device)

    in_channels = model.decode_head.classifier.in_channels
    model.decode_head.classifier = torch.nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    return model
    

def build_hugginface_models(cfg, device = "cuda"):
    model = load_model(cfg.path2pretrained, num_classes =  cfg.num_classes + 1, device = device)
    return model








