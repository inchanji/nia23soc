import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist



def cleanup():
	dist.destroy_process_group()
	torch.cuda.empty_cache()
	return True
