import cv2
import numpy as np
import json

path_img = 'res/ex.jpg'
path_json = 'res/ex.json'

# read image   
img = cv2.imread(path_img)

# read json 
with open(path_json) as f:
    data = json.load(f)

print(len(data['annotations']))
print(data['annotations'][0]['label'])
print(data['annotations'][0]['shape'])
print(data['annotations'][0]['points'])

for x,y in data['annotations'][0]['points']:
    print(x,y)


# create mask on image using x, y 
mask = np.zeros(img.shape[:2], dtype=np.uint8)
points = np.array(data['annotations'][0]['points'])
cv2.fillPoly(mask, [points], (255,255,255))

# apply mask on image
result = cv2.bitwise_and(img, img, mask=mask)

# save result
cv2.imwrite('res/mask.jpg', mask)