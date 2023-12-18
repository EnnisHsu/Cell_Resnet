import numpy as np
import os
import torchvision
from torchvision import transforms
from PIL import Image
import random

if __name__=="__main__":
    data_path='/home/cell/datasets/cell/images/03早幼粒/早幼粒细胞/'
    save_path='./newzaoyouli_trad/'
    if (not os.path.exists(save_path)):
        os.mkdir(save_path)
    transform=torchvision.transforms.Compose(
        [transforms.Resize([336,336]),
        transforms.CenterCrop([224,224]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((0,360)),
        ] )
    # set=torchvision.datasets.ImageFolder(data_path)
    img_list=os.listdir(data_path)
    for img_num in range(0,1000):
        num = random.randint(0,len(img_list)-1)
        (img_old)=Image.open(data_path+img_list[num])
        print(img_old)
        # img_old=Image.fromarray(img_std)
        img_new=transform(img_old)
        img_new.save(save_path+str(img_num)+'.jpg')
        print('successfully save img %s' % (str(num)+'.jpg'))
        

