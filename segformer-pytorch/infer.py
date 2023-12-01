from src import * 






def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--arch', type=str, default='pretrained/segformer-b1-finetuned-ade-512-512.pth', help='model architecture')
    parser.add_argument('--model', type=str, 
                            default='weights/segformer-b1-finetuned-ade-512-512-512px-bs64-cosannealwarm-cEloss-3chimg-tta-sc-wbg1.0-best_metric.pth', help='model path')
    parser.add_argument('--imgsize', type=int, default=512, help='image size')
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--imgs', type=str, default='eximgs', help='image folder')

    # parser.add_argument('--batchsize', type=int, default=8, help='batch size')
    # parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    # parser.add_argument('--is_grayscale', type=int, default=1, help='is grayscale')
    # parser.add_argument('--valid_tta', type=int, default=0, help='is grayscale')
    # parser.add_argument('--patchsize', type=int, default=512, help='patch size')
    # parser.add_argument('--patchstride', type=int, default=256, help='patch stride')
    # parser.add_argument('--patchthreshold', type=float, default=0.5, help='patch threshold')
    # parser.add_argument('--patchminarea', type=float, default=0.0001, help='patch min area')
    # parser.add_argument('--patchmaxarea', type=float, default=0.5, help='patch max area')
    # parser.add_argument('--patchminwidth', type=float, default=0.1, help='patch min width')
    # parser.add_argument('--patchminheight', type=float, default=0.1, help='patch min height')
    # parser.add_argument('--patchminratio', type=float, default=0.1, help='patch min ratio')

    return parser.parse_args()





if __name__ == '__main__':
    from glob import glob
    args    = parse_args()

    # load model
    model = build_hugginface_models_v2(args.arch, num_classes = 11, device = f"cuda:{args.device}")

    model.load_state_dict(torch.load(args.model, map_location=f"cuda:{args.device}"))
    model.to(f"cuda:{args.device}")
    model.eval()


    # load data
    img_paths = glob(f"{args.imgs}/*")

    # inference

    for path in img_paths:
        print(path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (args.imgsize, args.imgsize))
        img = img.astype(np.float32)

        img = (img - MEANPIXVAL) / STDPIXVAL

        img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(f"cuda:{args.device}")
        with torch.no_grad():
            pred = model(img).logits
            pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    
        perd_colors = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

        for key in colors_dict.keys():
            print(key)
            perd_colors[pred== key] = colors_dict[key]

        # pred = cv2.resize(pred, (512, 512))

        cv2.imwrite(f"eximgs/{path.split('/')[-1].split('.')[0]}_pred.png", perd_colors)
        


        



        

    


