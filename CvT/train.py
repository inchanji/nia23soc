from src import *

def train(config):

    os.makedirs(f"{os.getcwd()}/weights/", exist_ok = True)

    path_train 	= f"{config.data_root}/CvT/train.csv"
    path_valid 	= f"{config.data_root}/CvT/valid.csv"    
    path_test 	= f"{config.data_root}/CvT/test.csv"

    train		= pd.read_csv(path_train)
    valid		= pd.read_csv(path_valid)
    test		= pd.read_csv(path_test)    

    # the number of classes
    config.num_classes = len(train['label'].unique())

    seed_everything(config.seed)
    set_proc_name(config, "nia23soc-train-" + config.model_arch)

    model_spec	= get_modelname_ext(config)
    device = select_device(config.device)

    train_loader = prepare_dataloader(train, config, is_training = True)
    valid_loader = prepare_dataloader(valid, config, is_training = False)
    test_loader  = prepare_dataloader(test, config, is_training = False)



    # print train and valid dataset
    print(f"train dataset: {len(train_loader.dataset)}")
    print(f"valid dataset: {len(valid_loader.dataset)}")

    for img, label in train_loader:
        print(img.shape)
        print(label.shape)
        break


    # model = build_model(config.model_yaml)
    
    # # load pretrained weights
    # if config.path2pretrained:
    #     model.load_state_dict(torch.load(config.path2pretrained))

    # # update the number of classes
    # model.head = torch.nn.Linear(model.head.in_features, config.num_classes)

    # print(model)
