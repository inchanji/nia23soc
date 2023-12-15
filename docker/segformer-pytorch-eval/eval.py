from src import *
import importlib
from collections import OrderedDict
	
	
def eval(config):
	print("evalutation")
	classinfo 		= classInfo(include_normal = config.include_normal)
	class_names 	 = classinfo.get_class_names()
	num_classes 	 = classinfo.num_classes

	model_spec	= get_modelname_ext(config)
	dir2save 	= f"outputs/{config.expName}/weights"

	model_path_best_metric 	= '{}/{}-{}-{}.pth'.format(dir2save, config.model_arch, model_spec, "best_metric")


	path_test 		= f"{config.data_root}/SegFormer/test.csv"    
	test			= pd.read_csv(path_test)
	# shuffle test
	test = test.sample(frac=1).reset_index(drop=True)

	total_test 		= len(test)
	test_loader  	= prepare_dataloader(test,  config, is_training = False)
	
	device = torch.device('cpu')
	model = build_hugginface_models(config, device = 'cpu')
	
	# load model
	state_dict 		= torch.load(model_path_best_metric, map_location=device)
	new_state_dict  = OrderedDict()

	# remove 'module.' from the key
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	model.load_state_dict(new_state_dict)

	
	model.to(device)
	model.eval()
	print("loading model from", model_path_best_metric)

	print(model)

	seed_everything(config.seed)
	set_proc_name(config, "nia23soc-test-" + config.model_arch)

	with torch.no_grad():
		_ = evaluate(0,
			   		config, 
			   		model, 
					None, 
					test_loader,
					device,
					threshold = 0.5,
					scheduler = None,
					schd_loss_update = False,
					wandb = None,
					visualize = False,
					max_uploads = 64,
					taskname = 'test'
					)
					   


def parse_args():
	parser = argparse.ArgumentParser(description='Train a segmentor')
	parser.add_argument('--config', 
						default='configs.segformer-b1-finetuned-ade-512-512', # segformer-b1-finetuned-ade-512-512 segformer-b2-finetuned-ade-512-512
						help='train config file path')

	return parser.parse_args()




if __name__ == "__main__":
	print("python3 eval.py")
	args    = parse_args()
	module  = importlib.import_module(args.config)
	CFG     = getattr(module, 'Config')

	eval(CFG)

