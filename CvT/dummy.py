from lib.models import build_model, build_model_v2
from config import config, update_config
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str, 
                        default='config/cvt-13-224x224.yaml')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    # update_config(config, args)
    # print(config)

    model = build_model_v2(args.cfg)
    
    

    print(config)
    model.to(torch.device('cuda'))
    print(model)

    input_tensor = torch.randn(1, 3, 512, 512).to(torch.device('cuda'))
    output = model(input_tensor)
    print(output.shape)


# args = {'cfg': 'config/cvt-13-224x224.yaml'}
# update_config(config, args)
# print(config)

# model = build_model(config)
# print(model)