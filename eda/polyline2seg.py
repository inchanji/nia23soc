from shapely.geometry import LineString, Polygon
import json
import numpy as np
import cv2

file_path = "./res/polyline.json" #JSON 산출물 파일 path

with open(file_path, 'r',encoding='utf-8') as file:
    annos = []
    data = json.load(file)
    # print(data)
    for item in data['image']['annotations']:               # task, video, image중 image의 annotations 항목으로 반복
        line = LineString(item['points'])                   # Line 좌표로 line 생성
        buffer_distance = item['px']                        # 만들어질 폴리곤의 px 값
        buffered_polygon = line.buffer(buffer_distance)     # 폴리곤으로 변환 완료
        # print(dir(buffered_polygon))
        item['points'] = list(buffered_polygon.exterior.coords)
        annos.append(item)
    # print(annos)
    # with open("./line_to_poly_exam_result.json",'w',encoding='utf-8') as f:
    #     f.write(json.dumps(annos,ensure_ascii=False))

mask   = np.zeros((1080,1920), dtype=np.uint8)

for item in annos:
    points = np.array(item['points'], np.int32)
    cv2.fillPoly(mask, [points], 255)

cv2.imwrite("./res/polyline2seg.png", mask)