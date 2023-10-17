import cv2
import json
import numpy as np 
import matplotlib.pyplot as plt 

label2id = {
    "crack":1,
    "reticular crack":2,
    "detachment":3,
    "spalling":4,
    "efflorescene":5,
    "leak":6,
    "rebar":7,
    "material separation":8,
    "exhilaration":9,
    "damage":10
}
id2label = {v: k for k, v in label2id.items()}

def poly2mask(path_json, path_img, out_dir='./nia23soc_local/mask/', save_dir=None):
    img = cv2.imread(path_img)

    # read json 
    with open(path_json) as f:
        data = json.load(f)['image']

    fn_mask = data['name'].replace(".jpg", "_mask.png")

    # create mask on image using x, y 
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for anno in data['annotations']:
        points = np.array(anno['points'])[:,::-1]
        points[:,0] = 1080 - points[:,0] # flip x-axis
        label_type = anno['label']
        expanded = expand_polygon(points, 1)
        expanded = np.array(expanded).astype(int)
        cv2.fillPoly(mask, [expanded], label2id[label_type])
    # apply mask on image
    #mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    # save result
    cv2.imwrite(out_dir + fn_mask, mask)
    if save_dir:
        img[:,:,2] = img[:,:,2] * (1-mask)
        cv2.imwrite(save_dir + fn_mask, img)

def expand_polygon(vertices, offset):
    expanded_vertices = []

    # Compute the outward normals for each edge
    normals = []
    for i in range(len(vertices)):
        p0 = vertices[i]
        p1 = vertices[(i+1) % len(vertices)]
        
        edge_dir = [p1[0] - p0[0], p1[1] - p0[1]]
        normal = [-edge_dir[1], edge_dir[0]]
        len_normal = (normal[0]**2 + normal[1]**2)**0.5
        normalized_normal = [normal[0]/len_normal, normal[1]/len_normal]
        normals.append(normalized_normal)

    # Expand each vertex
    for i in range(len(vertices)):
        prev_normal = normals[i-1]
        current_normal = normals[i]

        avg_normal = [(prev_normal[0] + current_normal[0])/2, (prev_normal[1] + current_normal[1])/2]
        expanded_vertex = [vertices[i][0] + avg_normal[0]*offset, vertices[i][1] + avg_normal[1]*offset]
        expanded_vertices.append(expanded_vertex)

    # You would still need to handle the intersection of expanded edges, especially for concave corners
    # This is a more complex operation which requires line-line intersection checks

    return expanded_vertices

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)