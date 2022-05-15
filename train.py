import argparse
import numpy as np
import os
#import matplotlib.pyplot as plt
import time
# import cv2

import torch
import torchmetrics
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from imblearn.over_sampling import RandomOverSampler

import resnet
import googlenet
import datasets


def train(start_epoch,epochtimes):
    global best_acc
    timestd = time.time()
    
    writer=SummaryWriter('./log/Resnet-4:1-baseline'+str(time.ctime()),10)
    for epoch in range(start_epoch,epochtimes):
        running_loss = 0.0
        net.train()
        # print('epoch:{}'.format(epoch))
        for i,data in enumerate(trainloader,0):
            inputs,labels = data
            # print(inputs,labels)
            inputs,labels=Variable(inputs).to(device),Variable(labels).to(device)
            # print(labels.size())
            optimizer.zero_grad()
            outputs=net(inputs).to(device)
            loss=criterion(outputs,labels)
            #train_acc(outputs,labels)
            loss.backward()
            optimizer.step()
            # print("loss:{}".format(loss))
            running_loss += loss.data
        #acc=train_acc.compute()
        
        if epoch%10==0:
            torch.save({
                    'epoch': epoch ,
                    'state_dict':net.state_dict(),
                    'best_acc':best_acc,
                    'optimizer':optimizer.state_dict(),
                    },'./backup/resnet_auto.pkl')
            print('autosave resnet_auto.pkl')
        
        net.eval()
        acc=0.0
        class_correct = list(0. for i in range(14))
        class_total = list(0. for i in range(14))
        for i,data in enumerate(testloader,0):
            
            inputs,labels=data
            inputs,labels=Variable(inputs).to(device),Variable(labels).to(device)

            outputs=net(inputs).to(device)
            train_acc(outputs,labels)
            _,predicted = torch.max(outputs,1)
            c = (predicted == labels).squeeze()
            for j in range(labels.numel()):
                
                label = labels[j]
                class_correct[label] += c[j].item()
                class_total[label] +=1

        acc=train_acc.compute()
        best_acc=max(acc,best_acc)
        print('[ epoch:%d/%d ] time:%d s loss:%.3f acc:%.3f best_acc:%.3f' % (epoch+1,epochtimes, time.time()-timestd,running_loss/len(trainloader),acc*100,best_acc*100))
        #loss_list.append(running_loss/len(trainloader))
        #accuracy_list.append(acc)
        writer.add_scalar('Loss',loss,epoch)
        writer.add_scalar('Accuracy',acc,epoch)
        for i in range(14):
            if (class_total[i]!=0):
                print('Accuracy of %5s(%d) : %.3f %%' % (classes[i],class_total[i],100*class_correct[i]/class_total[i]))
            else:
                print('Accuracy of %5s(%d) : %.3f %%' % (classes[i],class_total[i],-1))
        running_loss=0
        train_acc.reset()
    print('Finished Training')
    torch.save({
        'epoch': epoch ,
        'state_dict':net.state_dict(),
        'best_acc':best_acc,
        'optimizer':optimizer.state_dict(),
        },'./backup/resnet.pkl')


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(description='CellResnet')
    #parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu',action='store_true', default=False,help='use gpu or not')
    parser.add_argument('-epoch',type=int,default=100,help='epoch times')
    parser.add_argument('-start_epoch',type=int,default=0,help='start epoch times')
    parser.add_argument('-b', type=int, default=32, help='batch size for dataloader') 
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--resume_path', default='./backup/resnet_auto.pkl', type=str, metavar='PATH',
                    help='path to latest checkpoint ')
    args = parser.parse_args()
    

    device=torch.device('cuda'if torch.cuda.is_available() and args.gpu else 'cpu')
    print('program run on {}'.format(device))

    net=models.resnet50(pretrained=True)
    net.to(device)

    optimizer=optim.SGD(net.parameters(),args.lr,momentum=0.9)
    criterion=nn.CrossEntropyLoss()
    train_acc=torchmetrics.Accuracy().to(device)
    best_acc=0.0

    if args.resume:
        if os.path.isfile(args.resume_path):
            print('loading checkopint {}'.format(args.resume_path))
            resume_point=torch.load(args.resume_path)
            print('check point data:{}'.format(resume_point['epoch']))
            args.start_epoch = resume_point['epoch']
            net.load_state_dict(resume_point['state_dict'])
            optimizer.load_state_dict(resume_point['optimizer'])
            best_acc = resume_point['best_acc']
            print('loaded checkpoint {}(epoch: {})'.format(args.resume_path,args.start_epoch))
        else:
            print('no checkpoint found at {}'.format(args.resume_path))
    
    transform_train=torchvision.transforms.Compose(
        [transforms.Resize([216,216]),
        transforms.RandomCrop([216,216],padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),] )
    # set=torchvision.datasets.ImageFolder('./dataset/',transform_train)
    classes=('01单核系','02原粒','03早幼粒','04中幼粒','05晚幼粒','06杆状核','07分叶核','08其他粒系','09其他红系','10中幼红','11晚幼红','12原淋细胞','13成熟淋巴细胞','14其他淋巴细胞')
    # print(set.imgs[8000][0])
    # train_size=int(0.8*len(set))
    # test_size=int(len(set)-train_size)
    # trainset,testset= torch.utils.data.random_split(dataset=set,lengths=[train_size,test_size])
    
    # print('Trainset')
    # ls=-1
    # cnt=0
    # for i in range(train_size):
    #     #print(path,labels)
    #     if (trainset.imgs[i][1]==ls):
    #         cnt+=1
    #     else:
    #         print(classes[trainset.imgs[i][1]+1],cnt)
    #         cnt=1
    #         ls=trainset.imgs[i][1]
    
    trainset = datasets.BatchDataset(transform_train)
    testset = datasets.RandomDataset(transform_train)
    print('loaded trainset:{} testset:{}'.format(len(trainset),len(testset)))
    trainloader=DataLoader(trainset,args.b,shuffle=True,num_workers=16,pin_memory=True)
    testloader=DataLoader(testset,args.b,shuffle=True,num_workers=16,pin_memory=True)

    print('Start Training')
    train(args.start_epoch,args.epoch)


                                                                                                                    

     

