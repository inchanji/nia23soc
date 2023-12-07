import numpy as np
import cv2
from glob import glob


imgsize = 512

src_dir = "/home/inchanji/data/nia23soc/원천데이터/지상시설물/도로교량/상부구조"
tgt_dir = "./test_imgs"

paths = glob(f"{src_dir}/*")
for i, path in enumerate(paths):
    print(path)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # split image to 512x512
    h, w, _ = img.shape
    fname = path.split('/')[-1].split('.')[0]

    for y in range(0, h, imgsize):
        for x in range(0, w, imgsize):
            img_ = img[y:y+imgsize, x:x+imgsize]
            cv2.imwrite(f"{tgt_dir}/{fname}_{y}_{x}.png", img_)

    

    if i > 10:
        break

    
    # 

