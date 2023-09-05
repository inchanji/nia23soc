from .registry import model_entrypoints
from .registry import is_model
import yaml
from yacs.config import CfgNode as CN

def build_model(cfg_file):
    config = CN()
    config.set_new_allowed(True)
    config.merge_from_file(cfg_file)

    # print(config.MODEL.SPEC)

    model_name = config.MODEL.NAME
    if not is_model(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    return model_entrypoints(model_name)(config)
