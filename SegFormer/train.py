from src import *
train_val_test = ['train.csv', 'valid.csv', 'test.csv']

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train segmentation network')

    parser.add_argument('--root_path',
                        help='root path to the dataset',
                        type=str, 
                        default='../dataset/SegFormer')
    
    args = parser.parse_args()
    return args





if __name__ == "__main__":
    device = 'cuda' if cuda.is_available() else 'cpu'
    cuda.empty_cache()
    args        = parse_args()
    root_path   = args.root_path

    print("Loading dataset...")

    train = pd.read_csv(f"{root_path}/train.csv")
    valid = pd.read_csv(f"{root_path}/valid.csv")
    test  = pd.read_csv(f"{root_path}/test.csv")

    print("Dataset loaded.")
    print(train.head())
    print(valid.head())
    print(test.head())


    train_dataset = cast_dataset(train['img_path'], train['label'])
    valid_dataset = cast_dataset(valid['img_path'], valid['label'])
    test_dataset  = cast_dataset(test['img_path'], test['label'])


    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset, # optional??
        "validation": valid_dataset,
    }
    )

    print(dataset)


    pretrained_model_name = "nvidia/mit-b0" 
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        id2label=id2label,
        label2id=label2id
    )
    model = model.to(device)    

    print(model)



    train_dataset.set_transform(train_transforms)
    valid_dataset.set_transform(val_transforms)
    test_dataset.set_transform(val_transforms)
    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )    


    trainer.train()