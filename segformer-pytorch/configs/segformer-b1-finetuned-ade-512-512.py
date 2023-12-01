from configs.base import base

Config 		= base.CFG()


Config.model_arch 		= "segformer-b1-finetuned-ade-512-512"
Config.path2pretrained  = f"pretrained/{Config.model_arch}.pth"

Config.data_root        = '/home/inchanji/workspace/nia23soc/dataset'

Config.apex             = True
Config.device     		= 'cuda:0' #'mps' #'cuda:0'
Config.imgsize   		= 256

Config.epochs 	  		= 150
Config.patience   		= 10

Config.num_classes      = 10

Config.enable_patch     = True
Config.include_normal   = True   
Config.w_normal         = 1.0


# Config.dataset    	= "RESIZED"

Config.scheduler  		= 'CosineAnnealingWarmRestarts'

Config.dtype        	= 'point-base'
Config.is_grayscale 	= False
Config.loss 			= 'cEloss'


Config.use_weight       = False

Config.train_bs        = 64
Config.valid_bs        = 64

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

Config.BCELoss 	    	= False
Config.gradual_increase_trainset = True

Config.debug 	  		= False

Config.finalize()