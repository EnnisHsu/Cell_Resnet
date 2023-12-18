import numpy as np
import os
import torchvision
from PIL import Image

# data_path='/home/cell/datasets/cell/images/03早幼粒/newzaoyouli_trad/'

def image_compose():
    data_path='/home/cell/datasets/cell/images/03早幼粒/newzaoyouli_trad/'
    imgh=224
    imgc=224
    row=0
    col=0
    big_image=Image.new('RGB',(10*imgh,10*imgc))
    for i in range(1,101):
        img=Image.open(data_path+str(i)+'.jpg')
        big_image.paste(img,(row*imgh,col*imgc))
        print('successfully paste image No.%d' % (i))
        col+=1
        if (col==10):
            row+=1
            col=0
    big_image=big_image.resize([512,512])
    big_image.save(data_path+'big_image.jpg')
    print('successfully save big image to %s' % (data_path+'big_image.jpg'))

image_compose()