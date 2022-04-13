import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

def default_loader(path):
    try:
        img = Image.open(path).convert('RGB')
    except:
        with open('read_error.txt','a') as fid:
            fid.write(path+'\n')
        return Image.new('RGB',(224,224),'white')
    return img

class BatchDataset(Dataset):
    def __init__(self,transform=None,dataloader=default_loader):
        self.transform = transform
        self.dataloader = dataloader
        
        with open('/home/cell/datasets/cell/train.txt','r') as fid:
            self.imglist = fid.readlines()
        
        self.labels = []
        for line in self.imglist:
            img_path,label = line.strip().split()
            self.labels.append(int(label))
        self.labels = np.array(self.labels)

    def __getitem__(self,index):
        image_name,label = self.imglist[index].strip().split()
        image_path = image_name[1:]
        image_path="/home/cell/datasets"+image_path

        img = self.dataloader(image_path)
        img = self.transform(img)
        label = int(label)-1
        # label = torch.LongTensor([label])

        return [img,label]

    def __len__(self):
        return len(self.imglist)

class RandomDataset(Dataset):
    def __init__(self, transform=None, dataloader=default_loader):
        self.transform = transform
        self.dataloader = dataloader

        with open('/home/cell/datasets/cell/val.txt', 'r') as fid:
            self.imglist = fid.readlines()

    def __getitem__(self, index):
        image_name, label = self.imglist[index].strip().split()
        image_path = image_name
        image_path=image_path[1:]
        image_path="/home/cell/datasets"+image_path
        img = self.dataloader(image_path)
        img = self.transform(img)
        label = int(label)-1
        # label = torch.LongTensor([label])

        return [img, label]


    def __len__(self):
        return len(self.imglist)