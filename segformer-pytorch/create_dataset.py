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

line_w_mult = 2.0
min_width   = 2.0

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

train_val_test = [ 'valid.csv', 'test.csv', 'train.csv']
# train_val_test_inc_normal = [ 'valid.csv', 'test.csv', 'train.csv']

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

        x_centers   = []
        y_centers   = []
        classes     = []
        img_w       = []
        img_h       = []


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

            # meshgrid
            x, y = np.meshgrid(np.arange(col), np.arange(row))

            mask  = np.zeros((row, col), dtype=np.uint8)            



            with open(path2label, 'r',encoding='utf-8') as file:
                is_line     = []
                anns_seg    = []

                _classes = []

                data = json.load(file)

                # check if there is annotation
                if not 'annotations' in data['image']:
                    is_normal = True    
                    _class_str = '0'
                else:
                    is_normal = False    
                    # try:
                    if len(data['image']['annotations']) > 0:
                        for item in data['image']['annotations']:               # task, video, image중 image의 annotations 항목으로 반복
                            class_ = item['label']
                            try:
                                if classeID[class_] in [0, 1]:
                                    line = LineString(item['points'])                   # Line 좌표로 line 생성
                                    buffer_distance  = item['px']*line_w_mult            # 만들어질 폴리곤의 px 값, 실제 model 학습시에는 line_w_mult 배로 늘려서 학습
                                    buffer_distance  = max(buffer_distance, min_width)   # 최소값보다 작으면 최소값(min_width)으로 설정
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

                            if not classeID[class_] in classes:
                                _classes.append(classeID[class_]+1)

                            # unique classes
                            _classes = list(set(_classes))

                            # sort _classes 
                            _classes.sort()

                            # convert _classes to string
                            _class_str = ''
                            for _cls in _classes:
                                _class_str += f'{_cls},'
                            _class_str = _class_str[:-1]

                    else:
                        _class_str = '0'


            if is_normal:
                xcen = col/2
                ycen = row/2
                

            else:
                x_cen = np.mean(x[mask>0])
                y_cen = np.mean(y[mask>0])

            x_centers.append(x_cen)
            y_centers.append(y_cen)              

            classes.append(_class_str)

            img_w.append(col)
            img_h.append(row)
            
                        
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

            # print(len(img_paths), len(label_paths), len(x_centers), len(y_centers), len(classes))

            assert len(img_paths) == len(label_paths), "img_paths and label_paths are not same length: {} vs. {}".format(len(img_paths), len(label_paths))


        df_seg = pd.DataFrame({'img_path': img_paths, 
                                'label': label_paths, 
                                'class': classes, 
                                'x_cen': x_centers, 
                                'y_cen': y_centers,
                                'img_w': img_w,
                                'img_h': img_h})

        # save 
        df_seg.to_csv(f"{tgt_path}/{csv}", index=False)
        print(f"{csv} done")








