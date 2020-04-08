import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader
# from dataset.pre_processing import my_PreProc
import cv2
import random
# class_weight = torch.FloatTensor([0.25, 0.25, 0.25,1])
class_weight = torch.FloatTensor([0.0057471264, 0.0050251, 0.00884955752,1])

# mean = [0.611, 0.506, 0.54]
mean = [0.6127558736339982,0.5071148744673234,0.5406509545283443]

std = [0.13964046123851956,0.16156206296516235,0.165885041027991]

testmean = [0.6170943891910641,0.5133861905981716,0.545347489522038]
teststd = [0.14098655787705194,0.16313775003634445,0.16636559984060037]
class_color = [
    (128, 128, 128),
    (128, 0, 0),
    (192, 192, 128),
    (128, 64, 128),
]


def _make_dataset(dir, Gray):
    names = []
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('RGB.png'):
                if Gray:
                    fname = fname.replace('_RGB', '')
                else:
                    fname = fname
                path = os.path.join(root, fname)
                name = path.split('/')[-1].split('.')[0][:-4]
                # print(path)
                # print(name) 
                names.append(name)
                images.append(path)
    return images, names

class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic)#.long()
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            label = label.view(pic.size[1], pic.size[0], 1)
            label = label.transpose(0, 1).transpose(0, 2).squeeze().contiguous()#.long()
        return label

class CamVid(data.Dataset):
    def __init__(self, root, Gray, index_start = 0, index_end = 0, joint_transform=None, transform=None, target_transform=LabelToLongTensor(), loader=default_loader):
        self.root = root
        self.Gray = Gray
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.loader = loader
        # print('self.root', self.root)
        self.imgs_all, self.names_all = _make_dataset(self.root, self.Gray)
        self.imgs = self.imgs_all[index_start : index_end]
        self.names = self.names_all[index_start : index_end]
        print('len',index_start, index_end, len(self.imgs))
    def __getitem__(self, index):
        path = self.imgs[index]
        name = self.names[index]
        if self.Gray:
            img = self.loader(path).convert('L')
            target = Image.open(os.path.join(self.root, name + '_Tar.png'))
        else:
            img = self.loader(path)
            target = Image.open(os.path.join(self.root, name + '_Tar.png'))
        if self.joint_transform is not None:
            img, target = self.joint_transform([img, target])
        if self.transform is not None:
            img = self.transform(img)
        target = self.target_transform(target)/85
        # print('img',img.data.numpy().shape, np.max(img.data.numpy()))
        # print('target',name, target.shape, np.unique(target))
        return img, target, name
    def __len__(self):
        return len(self.imgs)
