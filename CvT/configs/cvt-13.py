from configs.base import base

Config 		= base.CFG()

Config.model_arch 		= "cvt-13-224x224"
Config.model_yaml 		= "configs/cvt-13-224x224.yaml"
Config.path2pretrained  = "pretrained/CvT-13-224x224-IN-1k.pth"

Config.data_root        = './dataset'


Config.device     		= 'cuda:0' #'mps' #'cuda:0'
Config.imgsize   		= 512
Config.debug 	  		= False
Config.epochs 	  		= 150
Config.patience   		= 10

# Config.dataset    	= "RESIZED"

Config.scheduler  		= 'CosineAnnealingWarmRestarts'

Config.dtype        	= 'point-base'
Config.is_grayscale 	= False
Config.loss 			= 'cross_entropy'


Config.weight_lm 		= True

Config.train_bs        = 4
Config.valid_bs        = 8

# wing loss params
Config.wingloss_w 		= 10
Config.wingloss_e 		= 0.5
Config.valid_tta 		= True
Config.OOF              = False

Config.do_fmix          = False
Config.do_fmix_v2       = False
Config.do_cutmix        = False
Config.do_tile_mix      = False
Config.aug_mix          = False

Config.finalize()
