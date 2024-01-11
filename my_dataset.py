from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_lpath: list, transform=None, split="train"):
        self.images_path = images_path
        self.images_lpath = images_lpath
        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        img_ref = Image.open(self.images_lpath[item])
        if img_ref.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))

        if self.split == "val":
            #For LOL, the size is 600 * 400. For PairL1.6K, the size is 224 * 224.
            img = img.resize((600, 400))
            img_ref = img_ref.resize((600, 400))

        if self.transform is not None:
            img, img_ref = self.transform(img, img_ref)
        return img, img_ref

    @staticmethod
    def collate_fn(batch):
        images, images_ref = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        images_ref = torch.stack(images_ref, dim=0)
        return images, images_ref
