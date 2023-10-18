from datasets import Dataset, DatasetDict, Image


def cast_dataset(image, label):
    dataset = Dataset.from_dict({"pixel_values": sorted(image),
                                 "label": sorted(label)})
    dataset = dataset.cast_column("pixel_values", Image())
    dataset = dataset.cast_column("label", Image())
    return dataset


# def create_dataset(image_paths, label_paths, test_size=0.2, valid_size=0.1, random_state=123):
#     img_train, img_rem, label_train, label_rem = train_test_split(image_paths, 
#                                                                     label_paths, 
#                                                                     test_size=test_size+valid_size, 
#                                                                     random_state=random_state)
    
#     img_valid, img_test, label_valid, label_test = train_test_split(img_rem, 
#                                                                     label_rem, 
#                                                                     test_size = 1 - valid_size/test_size, 
#                                                                     random_state=random_state)
    
#     train_dataset = cast_dataset(img_train, label_train)
#     test_dataset = cast_dataset(img_test, label_test)
#     valid_dataset = cast_dataset(img_valid, label_valid)

#     return train_dataset, test_dataset, valid_dataset
