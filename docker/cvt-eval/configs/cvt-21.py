from configs.base import base

Config 		= base.CFG()

Config.model_arch 		= "cvt-21-384x384"
Config.model_yaml 		= "configs/cvt-21-384x384.yaml"
Config.path2pretrained  = "pretrained/CvT-21-384x384-IN-22k.pth"

Config.data_root        = '../dataset'
Config.multiclass       = False

Config.apex             = True
Config.device     		= 'cpu' #'mps' #'cuda:0'
Config.imgsize   		= 512

Config.epochs 	  		= 500
Config.patience   		= 20
Config.num_classes      = 10
Config.include_normal   = True


# Config.dataset    	= "RESIZED"

Config.scheduler  		= 'CosineAnnealingWarmRestarts'

Config.dtype        	= 'point-base'
Config.is_grayscale 	= False
Config.loss 			= 'cEloss'


Config.use_weight       = False

Config.train_bs        = 16
Config.valid_bs        = 32

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

Config.debug 	  		= False

Config.finalize()
