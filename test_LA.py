import cv2
import os
import torch
from PIL import Image
from torchvision import transforms
from LLIEFormer_LA import LLIEFormer as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    root="./dataset/LOLdataset/eval15/low"
    assert os.path.exists(root), "file: '{}' dose not exist.".format(root)

    images_path = loadfiles(root=root)
    for index in range(len(images_path)):
        assert os.path.exists(images_path[index]), "file: '{}' dose not exist.".format(images_path[index])
    print("path checking complete!")
    print("confirmly find {} images for computing".format(len(images_path)))
    model = create_model().to(device)
    model_weight_path = "./weights/LOL_LA_model.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device)['model'])
    model.eval()
    for img_path in images_path:
        img = Image.open(img_path)
        img = data_transform(img).unsqueeze(0)
        with torch.no_grad():
            output = (model(img.to(device))).cpu()
        output = recover_img(output)
        output = output.squeeze(0)
        output = output.swapaxes(2, 1)
        outputpic = output.T.numpy()
        name=getnameindex(img_path)
        savepic(outputpic,name)

def savepic(outputpic,name):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]
    root = "./result/LOL_eval_LLIEFormer"
    if os.path.exists(root) is False:
        os.makedirs(root)
    path = root + "/{}.png".format(name)
    cv2.imwrite(path, outputpic)
    assert os.path.exists(path), "file: '{}' dose not exist.".format(path)
    print("complete compute {}.png and save".format(name))

def loadfiles(root):
    images_path = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    images = [os.path.join(root, i) for i in os.listdir(root)
              if os.path.splitext(i)[-1] in supported]
    for index in range(len(images)):
        img_path = images[index]
        images_path.append(img_path)
    print("find {} images for computing.".format(len(images_path)))
    return images_path

def recover_img(x):
    x = x * 0.5 + 0.5
    return x

def getnameindex(path):
    assert os.path.exists(path), "file: '{}' dose not exist.".format(path)
    label = path.split("\\")[-1].split(".")[0]
    return label

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
