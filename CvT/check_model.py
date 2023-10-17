from src import *

if __name__ == '__main__':
    model_yaml = 'configs/cvt-21-384x384.yaml'
    path2pretrained = 'pretrained/CvT-21-384x384-IN-22k.pth'

    model = build_model(model_yaml, path2pretrained, num_classes = 10)
    
    print(model)

    with torch.no_grad():
        model.eval()
        x = torch.randn(10, 3, 512, 512)
        y = model(x)
        print(y.shape)
        # print(y)