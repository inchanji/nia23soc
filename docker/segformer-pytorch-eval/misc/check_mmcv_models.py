import torch 


root_path = '/home/inchanji/workspace/nia23soc/segformer-mmcv/pretrained'
modelname = 'segformer.b0.1024x1024.city.160k.pth'

model = torch.load(f'{root_path}/{modelname}', map_location=torch.device('cpu'))

print (model.keys())

# print layer names
for key in model['state_dict'].keys():
    print (key)


    