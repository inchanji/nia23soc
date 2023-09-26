import cv2
import json
import numpy as np 
import matplotlib.pyplot as plt 

label2id = {
    "crack":1,
    "reticular crack":2,
    "detachment":3,
    "spalling":4,
    "efflorescence":5,
    "leak":6,
    "rebar":7,
    "material separation":8,
    "exhilartion":9,
    "damage":10
}
id2label = {v: k for k, v in label2id.items()}

def poly2mask(path_json, out_dir='./nia23soc_local/mask/'):
    path_img = path_json.replace("2.라벨링데이터", "1.원천데이터").replace(".json", ".jpg")

    img = cv2.imread(path_img)

    # read json 
    with open(path_json) as f:
        data = json.load(f)['image']

    fn_mask = data['name'].replace(".jpg", "_mask.png")

    # create mask on image using x, y 
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for anno in data['annotations']:
        points = np.array(anno['points'])
        label_type = anno['label']
        cv2.fillPoly(mask, [points], label2id[label_type]) # 혹시 filloply 한번에 여러개의 polygon을 그릴 수 있나?
    # apply mask on image
    #result = cv2.bitwise_and(img, img, mask=mask)

    # save result
    cv2.imwrite(out_dir + fn_mask, mask)


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)