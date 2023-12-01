from src import *
import importlib

def eval(config):
	print("evalutation")
	classinfo = classInfo(include_normal = config.include_normal)
	class_names 	 = classinfo.get_class_names()
	num_classes 	 = classinfo.num_classes

	model_spec	= get_modelname_ext(config)
	dir2save 	= f"outputs/{config.expName}/weights"

	# model_path_best_metric 	= '{}/{}-{}-{}.pth'.format(dir2save, config.model_arch, model_spec, "best_metric")
	model_path_best_metric 	= '{}/{}-{}-{}.pt'.format(dir2save, config.model_arch, model_spec, "best_metric-full")
	path_test 		= f"{config.data_root}/SegFormer/test.csv"    
	test			= pd.read_csv(path_test)
	total_test 		= len(test)
	test_loader  	= prepare_dataloader(test,  config, is_training = False)

	device = torch.device(config.device)
	# model = build_hugginface_models(config)
	# # load model
	# model.load_state_dict(torch.load(model_path_best_metric, map_location=device))
	model = torch.load(model_path_best_metric, map_location=device)
	model.to(device)
	model.eval()
	print("loading model from", model_path_best_metric)

	print(model)

	seed_everything(config.seed)
	set_proc_name(config, "nia23soc-test-" + config.model_arch)

	normalized_confusion_matrix 	= torch.zeros(num_classes, num_classes).to(device)
	normalized_confusion_matrix_v2 	= torch.zeros(num_classes, num_classes).to(device)
	normalized_confusion_matrix_i 	= torch.zeros(num_classes, num_classes).to(device)

	miou_v3 = [0.] * num_classes
	miou_v3_count = [0] * num_classes
	
	miou_v4_val = 0.
	miou_v4_count = 0
	fname = ''
	print("{:10s} {:35s} {:5s} {:5s}".format("i/tot","fname", "iou", "miou"))
	n = 0
	for step, (images, labels, image_path) in enumerate(test_loader):
		labels  = labels.to(device)
		with torch.no_grad():
			preds   = model(images.float().to(device)).logits

		preds_i		= torch.argmax(preds, 1)	# (B, C, H, W) to (B, H, W)
		labels_i 	= labels 				# (B, H, W)

		# update confusion matrix
		binount = torch.bincount( preds_i.reshape(-1).long() * num_classes + labels_i.long().reshape(-1), 
											minlength = num_classes**2
											).reshape(num_classes, num_classes)
		normalized_confusion_matrix_i_ = binount / (labels_i.reshape(-1).shape[0]) 
		normalized_confusion_matrix_v2 += normalized_confusion_matrix_i_


		n_images = 0
		for i, path in enumerate(image_path):
			_fname 		= path.split('/')[-1]
			pred_i 		= preds_i[i, :, :].unsqueeze(0)
			label_i 	= labels_i[i, :, :].unsqueeze(0)

			bincount 	= torch.bincount( pred_i.reshape(-1).long() * num_classes + label_i.long().reshape(-1),
										minlength = num_classes**2
										).reshape(num_classes, num_classes)
			
			# normalized_confusion_matrix_i = bincount / (label_i.reshape(-1).shape[0]) 
			

			if fname != _fname:
				if fname == '':
					fname = _fname
					continue

			
				iou_all = []
				for j in range(num_classes):
					iou = normalized_confusion_matrix_i[j, j] / (normalized_confusion_matrix_i[j, :].sum() + normalized_confusion_matrix_i[:, j].sum() - normalized_confusion_matrix_i[j, j])
					iou_all.append(iou.item())

				iou = np.nanmean(iou_all)
				miou_v4_val += iou 
				miou_v4_count += 1

				confusion_mat = normalized_confusion_matrix.detach().cpu().numpy()
				miou_all = []
				for j in range(num_classes):
					miou_all.append(confusion_mat[j, j] / (confusion_mat[j, :].sum() + confusion_mat[:, j].sum() - confusion_mat[j, j]) )

				miou = np.nanmean(miou_all)

				confusion_mat = normalized_confusion_matrix_v2.detach().cpu().numpy().transpose()
				miou_all = []
				for j in range(num_classes):
					miou_all.append(confusion_mat[j, j] / (confusion_mat[j, :].sum() + confusion_mat[:, j].sum() - confusion_mat[j, j]) )
				
				miou_v2 = np.nanmean(miou_all)


				for j in range(num_classes):
					if not np.isnan(iou_all[j]):
						miou_v3_count[j] += 1
						miou_v3[j] += iou_all[j]

				# miou_v3_val = 0 
				# for j in range(num_classes):
				# 	if miou_v3_count[j] > 0:
				# 		miou_v3_val += miou_v3[j] / miou_v3_count[j]

				# miou_v3_val /= np.sum(miou_v3_count)

				miou_v3_all = []
				for j in range(num_classes):
					if miou_v3_count[j] > 0:
						miou_v3_all.append(miou_v3[j] / miou_v3_count[j])
					else:
						miou_v3_all.append(np.nan)
				miou_v3_val = 0
				cnt = 0
				for j in range(num_classes):
					if not np.isnan(miou_v3_all[j]):
						miou_v3_val += miou_v3_all[j]
						cnt += 1
				miou_v3_val /= cnt

				print("iou_all", iou_all)
				# print("miou_all", miou_all)
				print("{:5d}/{:5d} {:35s} {:5.4f} {:5.4f} {:5.4f} {:5.4f} {:5.4f} - {:1d} image(s) combined".format(n, total_test, _fname, iou, miou, miou_v2, miou_v3_val, miou_v4_val/miou_v4_count, n_images))

				# new image 
				fname = _fname
				normalized_confusion_matrix_i[:, :] = bincount / (label_i.reshape(-1).shape[0]) 
				n+=1
				n_images = 1
			else:
				normalized_confusion_matrix_i 	+= bincount / (label_i.reshape(-1).shape[0]) 
				n_images += 1
			normalized_confusion_matrix += normalized_confusion_matrix_i 

	normalized_confusion_matrix = normalized_confusion_matrix.detach().cpu().numpy().transpose()
	normalized_confusion_matrix /= normalized_confusion_matrix.sum()				

	iou_all = []
	for i in range(num_classes):
		iou_all.append(normalized_confusion_matrix[i,i] / (normalized_confusion_matrix[i,:].sum() + normalized_confusion_matrix[:,i].sum() - normalized_confusion_matrix[i,i]))
	miou_all = np.nanmean(iou_all)

	print("iou_all", iou_all)
	print("miou_all", miou_all)			


			

	
def eval_v2(config):
	print("evalutation")
	classinfo = classInfo(include_normal = config.include_normal)
	class_names 	 = classinfo.get_class_names()
	num_classes 	 = classinfo.num_classes

	model_spec	= get_modelname_ext(config)
	dir2save 	= f"outputs/{config.expName}/weights"

	# model_path_best_metric 	= '{}/{}-{}-{}.pth'.format(dir2save, config.model_arch, model_spec, "best_metric")
	model_path_best_metric 	= '{}/{}-{}-{}.pt'.format(dir2save, config.model_arch, model_spec, "best_metric-full")
	path_test 		= f"{config.data_root}/SegFormer/test.csv"    
	test			= pd.read_csv(path_test)
	total_test 		= len(test)
	test_loader  	= prepare_dataloader(test,  config, is_training = False)

	device = torch.device(config.device)
	# model = build_hugginface_models(config)
	# # load model
	# model.load_state_dict(torch.load(model_path_best_metric, map_location=device))
	model = torch.load(model_path_best_metric, map_location=device)
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
	parser.add_argument('--work-dir', help='the dir to save logs and models')
	parser.add_argument(
		'--load-from', help='the checkpoint file to load weights from')
	parser.add_argument(
		'--resume-from', help='the checkpoint file to resume from')
	parser.add_argument(
		'--no-validate',
		action='store_true',
		help='whether not to evaluate the checkpoint during training')
	group_gpus = parser.add_mutually_exclusive_group()
	group_gpus.add_argument(
		'--gpus',
		type=int,
		help='number of gpus to use '
		'(only applicable to non-distributed training)')
	group_gpus.add_argument(
		'--gpu-ids',
		type=int,
		nargs='+',
		help='ids of gpus to use '
		'(only applicable to non-distributed training)')
	parser.add_argument('--seed', type=int, default=None, help='random seed')
	parser.add_argument(
		'--deterministic',
		action='store_true',
		help='whether to set deterministic options for CUDNN backend.')
	parser.add_argument(
		'--options', nargs='+', action=DictAction, help='custom options')
	parser.add_argument(
		'--launcher',
		choices=['none', 'pytorch', 'slurm', 'mpi'],
		default='none',
		help='job launcher')
	parser.add_argument('--local_rank', type=int, default=0)
	args = parser.parse_args()
	if 'LOCAL_RANK' not in os.environ:
		os.environ['LOCAL_RANK'] = str(args.local_rank)

	return args

	


if __name__ == "__main__":

	args    = parse_args()
	module  = importlib.import_module(args.config)
	CFG     = getattr(module, 'Config')


	eval_v2(CFG)
