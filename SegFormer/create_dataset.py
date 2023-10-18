import json
import os
from glob import glob
import pandas as pd
from tqdm import tqdm
import argparse
import cv2

from shapely.geometry import LineString, Polygon
from sklearn.model_selection import train_test_split

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




train_val_test = ['train.csv', 'valid.csv', 'test.csv']

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train classification network')

    parser.add_argument('--root_path',
                        help='root path to the dataset',
                        type=str, 
                        default='/home/disk2/nia23soc')
    
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    args        = parse_args()
    root_path   = args.root_path
    cvt_path    = f"{root_path}/CvT"
    tgt_path    = f"{root_path}/SegFormer"



    for csv in train_val_test:
        df = pd.read_csv(f"{cvt_path}/{csv}")
    
        print(f"processing {csv}")
        # get paths to images
        img_paths   = []
        label_paths = []

        pbar = tqdm(enumerate(df['img_path']), total=len(df['img_path']))


        for idx, imgpath in pbar:
            fname = imgpath.split('/')[-1]
            pbar.set_description(f"processing {fname}")

        # for idx, imgpath in enumerate(df['img_path']):
            # print(imgpath)
            path2label = imgpath.replace('원천데이터', '라벨링데이터').replace('.jpg', '.json')

            if not os.path.exists(path2label):
                print(f"{path2label} not exist")
                exit(0)

            row, col = cv2.imread(imgpath).shape[:2]
            mask  = np.zeros((row, col), dtype=np.uint8)                

            with open(path2label, 'r',encoding='utf-8') as file:
                is_line = []
                anns_seg = []

                data = json.load(file)
                # try:
                for item in data['image']['annotations']:               # task, video, image중 image의 annotations 항목으로 반복
                    class_ = item['label']
                    try:
                        if classeID[class_] in [0, 1]:
                            line = LineString(item['points'])                   # Line 좌표로 line 생성
                            buffer_distance = item['px']                        # 만들어질 폴리곤의 px 값
                            buffered_polygon = line.buffer(buffer_distance)     # 폴리곤으로 변환 완료
                            
                            item['points'] = list(buffered_polygon.exterior.coords)

                            points = np.array(item['points'], np.int32)
                            cv2.fillPoly(mask, [points], classeID[class_]+1)
                            # cv2.fillPoly(mask, [points], (255,255,255))
                        else:
                            points = np.array(item['points'])
                            cv2.fillPoly(mask, [points], classeID[class_]+1) 
                            # cv2.fillPoly(mask, [points], (255,255,255))
                    except:
                        pbar.set_description(f"annotation error {fname}")
                
                        
            path2seg = imgpath.split("원천데이터")[-1]
            path2seg = f"{tgt_path}/segmentation{path2seg}" 
            path2seg = path2seg.replace('.jpg', '_mask.png')

            # check if directory exists
            if not os.path.exists(os.path.dirname(path2seg)):
                os.makedirs(os.path.dirname(path2seg))

            # imgpath.replace('원천데이터', '라벨링데이터').replace('.jpg', '_mask.png')
            
            cv2.imwrite(path2seg, mask)

            # print(imgpath)
            # print(path2seg)

            img_paths.append(imgpath)
            label_paths.append(path2seg)    

            assert len(img_paths) == len(label_paths), "img_paths and label_paths are not same length: {} vs. {}".format(len(img_paths), len(label_paths))


        df_seg = pd.DataFrame({'img_path': img_paths, 'label': label_paths})

        # save 
        df_seg.to_csv(f"{tgt_path}/{csv}", index=False)
        print(f"{csv} done")








