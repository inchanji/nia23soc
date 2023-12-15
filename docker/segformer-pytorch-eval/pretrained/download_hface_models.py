import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

model_arch = ['b0', 'b1', 'b2', 'b3', 'b4']
db = ['ade', 'cityscapes']
imgsizes = [512, 768, 1024, 1280, 1536, 1792, 2048]


# segformer-b0-finetuned-cityscapes-1024-1024
for _model_arch in model_arch:
    for _db in db:
        for _imgsize in imgsizes:
            modelname = f"segformer-{_model_arch}-finetuned-{_db}-{_imgsize}-{_imgsize}"
            
            try:
                model = SegformerForSemanticSegmentation.from_pretrained(f"nvidia/{modelname}")

                torch.save(model, f"{modelname}.pth")
                print(f"Saved {modelname}.pth")
            except:
                print(f"No such model: {modelname}")
                



