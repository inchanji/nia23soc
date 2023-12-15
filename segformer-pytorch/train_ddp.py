from src import *
import importlib
import argparse


def train(config):
	dist.barrier()
	os.makedirs(f"{os.getcwd()}/weights/", exist_ok = True)

	path_train 	= f"{config.data_root}/SegFormer/train.csv"
	path_valid 	= f"{config.data_root}/SegFormer/valid.csv"    
	path_test 	= f"{config.data_root}/SegFormer/test.csv"    

	train		= pd.read_csv(path_train)
	valid		= pd.read_csv(path_valid)
	test		= pd.read_csv(path_test)

	train_loader = prepare_dataloader_ddp(train, config, is_training = True)
	valid_loader = prepare_dataloader(valid, config, is_training = False)
	test_loader  = prepare_dataloader(test,  config, is_training = False)
	dist.barrier()
	if config.rank in [-1, 0]:
		print('data loaded')



	device 	= select_device(f"cuda:{config.rank}")
	model 	= build_hugginface_models(config)
	dist.barrier()
	if config.rank in [-1, 0]:
		print('model built')
	model.to(device)
	model 	= DDP(model, device_ids = [config.rank])
	dist.barrier()
	if config.rank in [-1, 0]:
		print('model loaded')

	# print(model)
	# print(device)
	# model.to(device)

	seed_everything(config.seed)
	set_proc_name(config, "nia23soc-train-" + config.model_arch)

	model_spec	= get_modelname_ext(config)
	

	optimizer   = prepare_optimizer(config, model)
	scheduler 	= get_scheduler(config, optimizer, len(train_loader))

	weight 		= [1.0] * (config.num_classes+1) if config.include_normal else [1.0] * config.num_classes
	if config.include_normal: 
		weight[0] = config.w_normal
	
	loss_tr 	= get_loss_fn(config, device, valid = False, weight = torch.Tensor(weight).half().to(device))
	loss_val 	= get_loss_fn(config, device, valid = True, weight = torch.Tensor(weight).float().to(device))
	loss_test 	= get_loss_fn(config, device, valid = True, weight = torch.Tensor(weight).float().to(device))


	
	if config.rank in [-1, 0]:
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



	if config.rank in [-1, 0] and wandb and wandb.run is None and not config.debug:
		wandb_run = wandb.init(project 	= config.projectName, 
							   name 	= "{}-{}".format(config.model_arch, model_spec),
							   config 	= class2dict(config),
							   job_type = "train")

		wandb_table = wandb.Table(columns=["category", "id", "target", "prob", "image", "grad_cam_image"])	


	patience 		= 0
	better_model 	= False
	best_loss 	 	= np.inf
	best_metric 	= 0

	dir2save = f"outputs/{config.expName}/weights"
	if not os.path.exists(dir2save):
		os.makedirs(dir2save, exist_ok = True)

	model_path_best_loss  	= '{}/{}-{}-{}.pth'.format(dir2save, config.model_arch, model_spec, "best_loss")
	model_path_best_metric 	= '{}/{}-{}-{}.pth'.format(dir2save, config.model_arch, model_spec, "best_metric")

	for epoch in range(config.epochs):
		# train 
		if config.rank in [-1, 0]: 
			print(f"Epoch: {epoch}")
			print('train')

		if config.gradual_increase_trainset:
			inc 		= 0.1
			max_epoch 	= int(1./inc)
			frac 		=  inc * (epoch + 1)
			frac 		= min(frac, 1.0)
			train_ 		= train.sample(frac = frac)
			# train_loader = prepare_dataloader(train_, config, is_training = True)
			train_loader = prepare_dataloader_ddp(train, config, is_training = True)


		
		avg_train_loss, avg_train_metric = train_one_epoch_ddp(epoch,
															config,
														  	model, 
														  	loss_tr, 
														  	train_loader, 
														  	optimizer, 
														  	device, 
														  	scheduler = scheduler, 
														  	schd_batch_update = False,
														  	wandb = wandb
														  )			

		# train_one_epoch(model, train_loader, device, epoch, config)	
		dist.barrier()
		if config.rank in [-1, 0]:
			print('train done. validation start.')
		
		
		# validation
		if config.rank in [-1, 0]:
			print("do validation")
		with torch.no_grad():
			avg_val_loss, avg_val_metric = evaluate_ddp(epoch, 
														config,
														model, 
														loss_val, 
														valid_loader, 
														device, 
														scheduler = None, 
														schd_loss_update = False,
														wandb = wandb,
														taskname = 'val'
														)
		if config.rank in [-1, 0]:
			print("do test")
		with torch.no_grad():
			avg_test_loss, avg_test_metric = evaluate_ddp(epoch, 
														config,
														model, 
														loss_test, 
														test_loader, 
														device, 
														scheduler = None, 
														schd_loss_update = False,
														wandb = wandb,
														taskname='test'
			)

		if config.rank in [-1, 0] and wandb and not config.debug:
			wandb.log( {f"epoch": epoch + 1})		

		if config.rank in [-1, 0]:
			if better_model:  patience = 0
			else:  patience += 1

			if avg_val_loss < best_loss:
				best_loss = avg_val_loss
				torch.save(model.state_dict(), model_path_best_loss)
				print("save best loss model")
				patience = 0

			if avg_val_metric['mIoU_all'] > best_metric:
				best_metric = avg_val_metric['mIoU_all']
				torch.save(model.state_dict(), model_path_best_metric)
				print("save best metric model")
				patience = 0

				# save the entire model
				torch.save(model, model_path_best_metric.replace(".pth", "-full.pt"))

		dist.barrier()
		torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors


	del model, optimizer, train_loader, valid_loader, test_loader, scheduler#, scaler
	with torch.cuda.device(config.device):
		torch.cuda.empty_cache()

	if wandb is not None: wandb.run.finish() 		
	dist.destroy_process_group()




def parse_args():
	parser = argparse.ArgumentParser(description='Train a segmentor')
	parser.add_argument('--config', 
						default='configs.segformer-b1-finetuned-ade-512-512', # segformer-b1-finetuned-ade-512-512 segformer-b2-finetuned-ade-512-512
						help='train config file path')
	
	parser.add_argument('--nnodes', type=int, default=0) 			# the number of nodes(nodes = number of machines)
	parser.add_argument('--node_rank', type=int, default=0) 		# the rank of node(0, 1, 2, 3, ...)
	parser.add_argument('--nproc_per_node', type=int, default=1) 	# the number of processes per node
	parser.add_argument('--nthreads_per_worker', type=int, default=8) # the number of threads per process
	parser.add_argument('--master_addr', default='localhost') 		# master node address
	parser.add_argument('--master_port', default='12355') 			# master node port
	parser.add_argument('--backend', default='nccl') 				# communication backend
	parser.add_argument('--rank', default=0) 						# global rank of the process
	parser.add_argument('--world_size', default=-1) 					# total number of processes to be created
	parser.add_argument('--init_method', default='env://') 			# initialization method
	return parser.parse_args()


if __name__ == "__main__":

	opt     = parse_args()
	module  = importlib.import_module(opt.config)
	CFG     = getattr(module, 'Config')


	CFG.rank 		= int(os.environ['RANK']) # global rank of the process
	CFG.world_size 	= int(os.environ['WORLD_SIZE']) # total number of processes to be created
	
	# setup dist
	print("rank {} world_size {}".format(CFG.rank, CFG.world_size))
	# print(CFG.rank, CFG.world_size)

	
	os.environ["OMP_NUM_THREADS"] = str(opt.nthreads_per_worker)
	dist.init_process_group(backend 	= opt.backend, 
						 	init_method = opt.init_method, 
							rank 		= CFG.rank, 
							world_size 	= CFG.world_size)
	
	train(CFG)
	
	cleanup() # end dist process
	if CFG.rank in [-1, 0]:
		print('cleanup done.')
