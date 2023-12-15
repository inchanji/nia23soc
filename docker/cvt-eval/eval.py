from src import *
import importlib
from datetime import datetime
from collections import OrderedDict

config_paths = [  "configs.cvt-21" ]


def eval(config):
	ts 	 =  datetime.timestamp(datetime.now())
	print("timestamp: ", ts)	
	path_test 	= f"{config.data_root}/CvT/test.csv"
	dir2save 	= f"outputs/{config.expName}/weights"
	model_spec	= get_modelname_ext(config)
	threshold 	= 0.5
	model_path_best_metric 	= '{}/{}-{}-{}.pth'.format(dir2save, config.model_arch, model_spec, "best_metric")

	test		= pd.read_csv(path_test)
	# shuffle the test dataset
	test = test.sample(frac=1).reset_index(drop=True)

	print("test dataset: ", len(test))

	print("test value counts: ")
	print(test.label.value_counts())	
	test_loader  = prepare_dataloader(test, config, is_training = False)

	# device 		= select_device(config.device)
	device 		= select_device('cpu')
	model 		= build_model(config.model_yaml, config.path2pretrained, num_classes = config.num_classes, multiclass = config.multiclass, device = device)
	state_dict 	= torch.load(model_path_best_metric, map_location=device)

	# remove 'module.' from the key
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	model.load_state_dict(new_state_dict)

	model.to(device)	
	model.eval()
	print(model)

	seed_everything(config.seed)
	set_proc_name(config, "nia23soc-test-" + config.model_arch)


	with torch.no_grad():
		labels_all = []
		preds_all = []
		for step, (fnames, images, image_labels) in enumerate(test_loader):
			image_preds   = model(images.float().to(device))

			if config.multiclass:
				# make pred as one-hot-encoding
				preds 	 = torch.ge(torch.sigmoid(image_preds), threshold).float().detach().cpu().numpy()
				confs = torch.sigmoid(image_preds).detach().cpu().numpy()
			else:
				preds 	 = torch.argmax(image_preds, 1).detach().cpu().numpy()
				confs = torch.softmax(image_preds, 1).detach().cpu().numpy()
			
			labels = image_labels.detach().cpu().numpy()
			fnames = list(fnames)

			
			ts 	 =  datetime.timestamp(datetime.now())
			print("{:13s} {:35s} {:5s} {:5s}   {:5s}  {:7s}".format("ts", "fname", "label", "pred", "conf", "f1_score(all)"))
			for fname, label, pred, conf in zip(fnames, labels, preds, confs):
				label = np.argmax(label)
				# print(fname, label, pred, np.max(conf))
				labels_all.append(label)
				preds_all.append(pred)

				f1_pred = np.eye(config.num_classes)[preds_all].astype(np.float32)
				f1_true = np.eye(config.num_classes)[labels_all].astype(np.float32)
				true_positive = (f1_pred * f1_true).sum(axis=1)
				precision = true_positive / (f1_pred.sum(axis=1) + 1e-8)
				recall = true_positive / (f1_true.sum(axis=1) + 1e-8)

				f1_score = (2 * precision * recall / (precision + recall + 1e-8)).mean()
				
				ts 	 =  datetime.timestamp(datetime.now())
				print("{:13.2f} {:35s} {:5d} {:5d}   {:5.4f}  {:5.4f}".format(ts, fname, label, pred, np.max(conf), f1_score))
		ts 	 =  datetime.timestamp(datetime.now())
		print("timestamp: ", ts, 'the final f1_score: ', f1_score)
	print("done.")
	

	




if __name__ == "__main__":
	print("python3 eval.py")
	for config_path in config_paths:
		module 		= importlib.import_module(config_path)
		CFG 		= getattr(module, 'Config')
		eval(CFG)






