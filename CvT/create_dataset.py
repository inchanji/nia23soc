from glob import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import json
import numpy as np

classeID = {
    'crack': 0,                 # 균열
    'reticular crack': 1,       # 망상균열
    'detachment': 2,            # 박리
    'spalling': 3,              # 박락
    'efflorescene': 4,          # 벡태(typo)
    'efflorescence': 4,         # 벡태
    'leak': 5,                  # 누수
    'rebar': 6,                 # 철근노출
    'material separation': 7,   # 재료분리
    'exhilaration': 8,          # 들뜸
    'damage': 9,                # 파손
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train classification network')

    parser.add_argument('--root_path',
                        help='root path to the dataset',
                        type=str, 
                        default='/home/inchanji/workspace/nia23soc/dataset')

    args = parser.parse_args()

    return args



if __name__ == "__main__":

    args = parse_args()
    root_path = args.root_path

    list_not_exist = []
    # get paths to images 

    root2img = f"{root_path}/원천데이터"
    soc_locations = [ s.split('/')[-1] for s in glob(f"{root2img}/*") ]
    soc_types = []
    for loc in soc_locations:
        soc_types += [ s.split('/')[-1] for s in glob(f"{root2img}/{loc}/*") ]
    soc_types = list(set(soc_types))


    print("soc_locations: ", soc_locations)
    print("soc_types: ", soc_types)

    classes = []
    img_paths = []
    classes_unique = []

    num = 0

    for loc in soc_locations:
        for typ in soc_types:
            print(f"{root2img}/{loc}/{typ}")
            # img_paths += glob(f"{root2img}/{loc}/{typ}/*.jpg")
            for imgpath in glob(f"{root2img}/{loc}/{typ}/*.jpg"):
                
                if len(imgpath.split('/')[-1].split('_')) == 5:
                    fname = imgpath.split('/')[-1]
                    print("wrong file name: ", fname)
                    HEAD, YYYY, MM, DD, EXT = fname.split('_')
                    imgpath_new = f"{root2img}/{loc}/{typ}/{HEAD}_{YYYY}_{MM}_{DD}_001_{EXT}"

                    os.rename(imgpath, imgpath_new)
                    imgpath = imgpath_new

                path2label = imgpath.replace('원천데이터', '라벨링데이터').replace('.jpg', '.json')

                # check if the label file exists
                if not os.path.exists(path2label):
                    list_not_exist.append(path2label)
                    continue

                # print(imgpath)
                with open(path2label) as f:
                    data = json.load(f)
                
                one_hot = [0] * 10

                for i in range(len(data['image']['annotations'])):
                    class_ = data['image']['annotations'][i]['label']
                    one_hot[classeID[class_]] = 1

                one_hot_str = ''
                for i in range(len(one_hot)):
                    one_hot_str += str(one_hot[i])
                    if i != len(one_hot)-1:
                        one_hot_str += ' '

                # class_ = data['image']['annotations'][0]['label']
                # class_ = data['image']['annotations'][0]['labelNum']
                img_paths.append(imgpath)
                classes.append(one_hot_str)

                # multi-label classification
                unique = int(np.sum(np.array(one_hot) * np.array([0,10,20,30,40,50,60,70,80,90])))
                classes_unique.append(unique)

                print(imgpath)
                print(path2label, unique)                

                # classes_unique.append(one_hot_str)
                # classes.append(classeID[class_])

                num += 1
                # if num == 256:
                #     break

    # print(list_not_exist)
    # print(img_paths)

    df = pd.DataFrame({'img_path': img_paths, 'label': classes, 'label_unique': classes_unique})

    print("total set: ", len(df))
    print(df['label'].value_counts())    
    print(df['label_unique'].value_counts())    

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)#, stratify=df['label_unique'])
    df_valid, df_test = train_test_split(df_test, test_size=0.5, random_state=42)#, stratify=df_test['label_unique'])


    df_train.to_csv(f"{root_path}/CvT/train.csv", index=False)
    df_valid.to_csv(f"{root_path}/CvT/valid.csv", index=False)
    df_test.to_csv(f"{root_path}/CvT/test.csv", index=False)

    # show the number of images per class in train and test set
    print("train set: ", len(df_train))
    print(df_train['label'].value_counts())
    print("valid set:", len(df_valid))
    print(df_valid['label'].value_counts())
    print("test set:", len(df_test))
    print(df_test['label'].value_counts())

    

    
