from glob import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import json

classeID = {
    'crack': 0,
    'reticular crack': 1,
    'spalling': 2,
    'detachment': 3,
    'rebar': 4
}



def parse_args():
    parser = argparse.ArgumentParser(
        description='Train classification network')

    parser.add_argument('--root_path',
                        help='root path to the dataset',
                        type=str, 
                        default='/home/inchanji/aws/nia23soc_local')

    args = parser.parse_args()

    return args



if __name__ == "__main__":

    args = parse_args()
    root_path = args.root_path

    # get paths to images 

    root2img = f"{root_path}/1.원천데이터"
    soc_locations = [ s.split('/')[-1] for s in glob(f"{root2img}/*") ]
    soc_types = []
    for loc in soc_locations:
        soc_types += [ s.split('/')[-1] for s in glob(f"{root2img}/{loc}/*") ]
    soc_types = list(set(soc_types))


    print("soc_locations: ", soc_locations)
    print("soc_types: ", soc_types)

    classes = []
    img_paths = []
    for loc in soc_locations:
        for typ in soc_types:
            # img_paths += glob(f"{root2img}/{loc}/{typ}/*.jpg")
            for imgpath in glob(f"{root2img}/{loc}/{typ}/*.jpg"):
                path2label = imgpath.replace('1.원천데이터', '2.라벨링데이터').replace('.jpg', '.json')

                with open(path2label) as f:
                    data = json.load(f)
                
                class_ = data['image']['annotations'][0]['label']
                # class_ = data['image']['annotations'][0]['labelNum']
                img_paths.append(imgpath)
                classes.append(classeID[class_])

    df = pd.DataFrame({'img_path': img_paths, 'label': classes})
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    df_valid, df_test = train_test_split(df_test, test_size=0.5, random_state=42, stratify=df_test['label'])


    df_train.to_csv(f"{root_path}/CvT/train.csv", index=False)
    df_valid.to_csv(f"{root_path}/CvT/valid.csv", index=False)
    df_test.to_csv(f"{root_path}/CvT/test.csv", index=False)

    # show the number of images per class in train and test set
    print("train set")
    print(df_train['label'].value_counts())
    print("valid set")
    print(df_valid['label'].value_counts())
    print("test set")
    print(df_test['label'].value_counts())

    

    
