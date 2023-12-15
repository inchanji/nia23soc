from .registry import model_entrypoints
from .registry import is_model
import yaml
import torch
from yacs.config import CfgNode as CN


class CustomCvt(torch.nn.Module):
    def __init__(self, model, in_features = 384, num_classes = 10):
        super().__init__()
        self.model = model
        self.head1 = torch.nn.Linear(in_features, 1)
        self.head2 = torch.nn.Linear(in_features, 1)
        self.head3 = torch.nn.Linear(in_features, 1)
        self.head4 = torch.nn.Linear(in_features, 1)
        self.head5 = torch.nn.Linear(in_features, 1)
        self.head6 = torch.nn.Linear(in_features, 1)
        self.head7 = torch.nn.Linear(in_features, 1)
        self.head8 = torch.nn.Linear(in_features, 1)
        self.head9 = torch.nn.Linear(in_features, 1)
        self.head10= torch.nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.model(x)
        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)
        x4 = self.head4(x)
        x5 = self.head5(x)
        x6 = self.head6(x)
        x7 = self.head7(x)
        x8 = self.head8(x)
        x9 = self.head9(x)
        x10 = self.head10(x)

        return torch.cat([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], dim = -1)



def build_model(cfg_file, path2pretrained, num_classes = 10, multiclass = False, device = 'cuda:0'):
    config = CN()
    config.set_new_allowed(True)
    config.merge_from_file(cfg_file)

    # print(config.MODEL.SPEC)

    model_name = config.MODEL.NAME
    if not is_model(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    model =  model_entrypoints(model_name)(config).to(device)

    # load pretrained weights
    if path2pretrained:
        model.load_state_dict(torch.load(path2pretrained, map_location = device))

    if not multiclass:
        model.head =  torch.nn.Linear(model.head.in_features, num_classes)
        return model
    else:
        in_features = model.head.in_features
        model.head = torch.nn.Identity()
        custom_model = CustomCvt(model, in_features = in_features,  num_classes = num_classes)
        return custom_model





