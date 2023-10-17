from src import *

def train(config):

	os.makedirs(f"{os.getcwd()}/weights/", exist_ok = True)

	if config.OOF:
		os.makedirs(f"{os.getcwd()}/oof/", exist_ok = True)
		path_train_oof = f"{os.getcwd()}/oof/{config.model_arch}_train.csv"
		path_valid_oof = f"{os.getcwd()}/oof/{config.model_arch}_valid.csv"
		path_test_oof  = f"{os.getcwd()}/oof/{config.model_arch}_test.csv"


	path_train 	= f"{config.data_root}/CvT/train.csv"
	path_valid 	= f"{config.data_root}/CvT/valid.csv"    
	path_test 	= f"{config.data_root}/CvT/test.csv"

	train		= pd.read_csv(path_train)[:128]
	valid		= pd.read_csv(path_valid)[:64]
	test		= pd.read_csv(path_test)[:64]

	train_loader = prepare_dataloader(train, config, is_training = True)
	valid_loader = prepare_dataloader(valid, config, is_training = False)
	test_loader  = prepare_dataloader(test, config, is_training = False)


	model = build_model(config.model_yaml, config.path2pretrained, num_classes = config.num_classes, multiclass = config.multiclass)
	
	

	seed_everything(config.seed)
	set_proc_name(config, "nia23soc-train-" + config.model_arch)

	if config.use_weight:
		weight = train.label.value_counts().to_numpy()
		weight = np.max(weight) / weight
		weight = torch.Tensor(weight)
	else:
		weight = None		

	model_spec	= get_modelname_ext(config)
	device 		= select_device(config.device)
	optimizer   = prepare_optimizer(config, model)
	scheduler 	= get_scheduler(config, optimizer, len(train_loader))

	loss_tr 	= get_loss_fn(config, device, valid = False, weight = weight)
	loss_val 	= get_loss_fn(config, device, valid = True)
	loss_test 	= get_loss_fn(config, device, valid = True)

	model.to(device)

	print("---------------------------------")
	print(f"> model architecture: {config.expName}")
	print(f"> device            : {device}\n")	

	# print(f"train dataset: {len(train_loader.dataset)}")
	# print(f"valid dataset: {len(valid_loader.dataset)}")
	# print(f"test dataset: {len(test_loader.dataset)}")

	print("> train dataset")
	print(train.head(5))
	print("> valid dataset")
	print(valid.head(5))
	print("> test dataset")
	print(test.head(5))

	print("\n> model architecture")
	print(model)
	print("\n> train optimizer    : ", optimizer)
	print("> train scheduler    : ", scheduler)
	print("> train loss         : ", loss_tr)
	print("> val loss           : ", loss_val)

	# disply parameters 
	print('> train params')
	display_cfg(config)

	print('\n> train transforms')
	for trans in get_train_transforms():
		if str(trans).split('(')[0] == 'OneOf':
			print(str(trans).split('(')[0])
			for subset in trans:
				print(" ", str(subset).split('(')[0])
		else:
			print(str(trans).split('(')[0])

	print('\n> valid transforms')
	for trans in get_valid_transforms():
		print(str(trans).split('(')[0])
	print('*************************')	

	
	
	if wandb and wandb.run is None :
		wandb_run = wandb.init(project 	= config.projectName, 
							   name 	= "{}-{}".format(config.model_arch, model_spec),
							   config 	= class2dict(config),
							   job_type = "train")

		wandb_table = wandb.Table(columns=["category", "id", "target", "prob", "image", "grad_cam_image"])	
	

	patience 		= 0
	better_model 	= False
	best_loss 	 	= np.inf
	best_metric 	= 0
	best_acc 		= 0

	model_path_best_loss  	= 'weights/{}-{}-{}.pth'.format(config.model_arch, model_spec, "best_loss")
	model_path_best_metric 	= 'weights/{}-{}-{}.pth'.format(config.model_arch, model_spec, "best_metric")
	model_path_best_acc 	= 'weights/{}-{}-{}.pth'.format(config.model_arch, model_spec, "best_acc")

	for epoch in range(config.epochs):
		avg_train_loss, avg_train_metric = train_one_epoch(epoch, 
															config,
															model, 
															loss_tr, 
															optimizer, 
															train_loader, 
															device, 
															scheduler 			= scheduler, 
															schd_batch_update 	= False,
															wandb 				= wandb
															)		
		# validation
		with torch.no_grad():
			avg_val_loss, avg_val_acc, avg_val_metric = valid_one_epoch(epoch, 
														config,
														model, 
														loss_val, 
														valid_loader, 
														device, 
														scheduler = None, 
														schd_loss_update = False,
														wandb = wandb
														)
		
		# test
		with torch.no_grad():
			avg_test_loss, avg_test_acc, avg_test_metric = valid_one_epoch(epoch, 
															config,
															model, 
															loss_test, 
															test_loader, 
															device, 
															scheduler = None, 
															schd_loss_update = False,
															wandb = wandb
															)
			
		if wandb:
			wandb.log( {f"epoch": epoch + 1, 
						f"avg_train_loss": avg_train_loss, 
						f"avg_train_f1": avg_train_metric, 

						f"avg_val_loss": avg_val_loss,
						f"avg_val_f1": avg_val_metric,
						f"avg_val_acc": avg_val_acc,

						f"avg_test_loss": avg_test_loss,
						f"avg_test_f1": avg_test_metric,
						f"avg_test_acc": avg_test_acc,
						})		
		
		if better_model:  patience = 0
		else:  patience += 1

		print("\n")
		if avg_val_loss < best_loss:
			print("saving best loss...")
			torch.save(model.state_dict(), model_path_best_loss)
			best_loss = avg_val_loss
			better_model 	= True

		if avg_val_metric > best_metric:
			print("saving best metric...")
			torch.save(model.state_dict(), model_path_best_metric)
			best_metric = avg_val_metric
			better_model 	= True

		if avg_val_acc > best_acc:
			print("saving best acc...")
			torch.save(model.state_dict(), model_path_best_acc)
			best_acc 		= avg_val_acc
			better_model 	= True

		if patience > config.patience and not config.debug:
			print('>>> Early stopping')
			break


	del model, optimizer, train_loader, valid_loader, scheduler#, scaler
	with torch.cuda.device(config.device):
		torch.cuda.empty_cache()

	if wandb is not None: wandb.run.finish() 

