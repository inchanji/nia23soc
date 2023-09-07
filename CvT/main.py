import importlib
from train import train

config_paths = [ 
				#   "configs.cvt-13",
				"configs.cvt-21",
]



if __name__ == "__main__":

	for config_path in config_paths:
		module 		= importlib.import_module(config_path)
		CFG 		= getattr(module, 'Config')
		train(CFG)


