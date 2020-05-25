# -*- coding: utf-8 -*-
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
import time
from PIL import Image
import re
from scipy.ndimage.interpolation import rotate
import copy
from torch.utils.data import Dataset
import random
import cv2

import scipy
from shutil import copyfile
def img_crop(img):
    w,h = img.shape[1], img.shape[2]
    crop_size = np.min((w,h))//2
    crop_img = img[:,w//2 - crop_size:w//2+crop_size,h//2-crop_size:h//2+crop_size]
    return crop_img
def scaleRadius(img, scale):
    # print(img.shape)
    # print(int(img.shape[0]/2))
    x = img[int(img.shape[0] / 2), :, :].sum(1)
    # print(x.mean())
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    #print(s)
    return cv2.resize(img, (0, 0), fx=s, fy=s)
def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img_new_1 = clahe.apply(img[:, :, 0])
    img_new_2 = clahe.apply(img[:, :, 1])
    img_new_3 = clahe.apply(img[:, :, 2])
    img_merge = cv2.merge([img_new_1, img_new_2, img_new_3])
    return img_merge

def aremove(img, scale):
    crop = scale * 2 * 0.9
    mask = np.ones(img.shape)
    #cv2.circle(mask, (int(img.shape[1] / 2), int(img.shape[0] / 2)), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
    gauss = cv2.GaussianBlur(img, (0, 0), scale / 50)
    enhanced = cv2.addWeighted(img, 4, gauss, -4, 128) * mask + 128 * (1 - mask)
    return enhanced
def data_process(img):
    crop_img = img_crop(img)
    crop_size = 256
    #gray_img = rgb2gray(crop_img)
    #print(crop_img.shape)
    N_img = crop_img.transpose(1,2,0)
    #Radius_img = scaleRadius(N_img, 256)
    clahe_img = clahe(N_img.astype("uint8"))
    enhanced = aremove(clahe_img,crop_size)
    enhanced_img = cv2.resize(enhanced,(512,512),interpolation=cv2.INTER_LINEAR).transpose(2,0,1)
    #print(enhanced_img.shape)
    #cv2.imwrite("{}_{}_{}.png".format(patient_id,label,age),enhanced_img[0].transpose(1,2,0).astype(np.uint8))
    return enhanced_img.astype(np.uint8)
def changeColorSpace(enhanced_img):
    r = np.random.randint(0, 10)
    enhanced_img = enhanced_img.transpose(1,2,0)
    #print(enhanced_img.shape)
    if r > 3 and r < 7:
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2HSV)
    elif r >= 6:
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2LUV)
    enhanced_img = enhanced_img.transpose(2, 0, 1)
    return enhanced_img
def findDepestFolder(path,finallist,deepnum = 0):
    dir = os.listdir(path)
    if len(dir) == 0:
        return
    dirs = []
    for ForD in dir:
        if os.path.isdir(os.path.join(path,ForD)):
            dirs.append(os.path.join(path,ForD))
    if len(dirs) == 0:
        finallist.append(path)
    else:
        for d in dirs:
            dd = os.path.join(path,d)
            findDepestFolder(dd,finallist,deepnum+1)
def addNoise(img,wh,size):
    num = np.random.randint(1,100)
    location = np.random.randint(0,wh-size,(num,2))
    for l in location:
        if len(img.shape) == 3:
            img[l[0]:l[0]+size,l[1]:l[1]+size,:] = 0
        else:
            img[:,:,l[0]:l[0] + size, l[1]:l[1] + size] = 0
    return img
def addFlip(img):
    r = np.random.randint(0,10)
    if len(img.shape) == 3:
        if r>3 and r<=5:
            img = img[:,::-1,:]
        if r>5 and r<=7:
            img = img[:,:,::-1]
        if r>7:
            img = img[:,::-1,::-1]
    if len(img.shape) == 4:
        if r>3 and r<=5:
            img = img[:,::-1,:]
        if r>5 and r<=7:
            img = img[:,::,::-1]
        if r>7:
            img = img[:,::-1,::-1]
    return img
def jumpGetNegtive(dirlist,total,b,name,numSeries):
    if np.random.randint(0,10)<2:
        dirlist1 = [dirlist[len(dirlist)//2]]
    else:
        total_num = np.random.randint(2,numSeries)
        dirlist1 = dirlist[len(dirlist)//2 - total_num//2:len(dirlist)//2+total_num//2]
    total = len(dirlist1)
    b = 1
    return jumpGet(dirlist1,total,b,name,numSeries)
def jumpGet(dirlist,total,b,name,numSeries):
    retlist = []
    if total >b*numSeries:
        i = np.random.randint(0,total-b*numSeries)
    else:
        i = 0
    while i<total:
        if random.randint(1,100)>75:
            size = np.random.randint(3,7)
            retlist.append(addFlip(addNoise(dirlist[i],dirlist[i].shape[-1],size)))
        else:
            retlist.append(addFlip(dirlist[i]))
        i+=b
    #if len(retlist) == 0:
        #print(name,i,b,total,len(dirlist),len(retlist),retlist)
    if len(retlist) == 1:
        imgblock = retlist[0]
    else:
        imgblock = np.concatenate(retlist,axis=0)
    if imgblock.shape[0]>numSeries-1:
        imgblock = imgblock[:numSeries-1,...]
        padnum = 1
    else:
        padnum = numSeries-imgblock.shape[0]
    imgblock = np.pad(imgblock,pad_width=((0,padnum),(0,0),(0,0),(0,0)),mode='constant',constant_values=(0,0))
    if random.randint(0,100)>97:
        if not os.path.exists('./lookimages'):
            os.mkdir('./lookimages')
        for i in range(imgblock.shape[0]):
            cv2.imwrite(str(i)+'.png',imgblock[i,:,:,:].transpose(2,1,0))
    return imgblock
class DataBowl(Dataset):
    def __init__(self, phase='train', policy = None,wh = (448,448
                                                          ),numSeries = 10):
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.policy = policy
        self.wh = wh
        self.numSeries =numSeries
        self.path_test = ['/data/gaozb/NCP/clinicalTest']


        self.imageAllTrue = []
        self.imageAllFalse = []
        self.savepathtest = '/data/chenxiangru/covidData/clinicalTest_deal_withname'
        if not os.path.exists(self.savepathtest):
            os.mkdir(self.savepathtest)
        if phase == 'test':
            for f in self.test_positive:
                self.image_name = [os.path.join(self.path_positive, i) for i in os.listdir(self.path_positive)]
                for fd in self.image_name_positive:
                    fdlist = []
                    findDepestFolder(fd, fdlist)
                    for folder in fdlist:
                        if len(os.listdir(folder)) == 0:
                            continue
                        self.imageAllTrue.append([[os.path.join(folder, i) for i in sorted(os.listdir(folder))], 1.])
            for f in enumerate(self.test_negtive):
                self.image_name = [os.path.join(self.test_negtive, i) for i in os.listdir(self.path_positive)]
                for fd in self.image_name_positive:
                    fdlist = []
                    findDepestFolder(fd, fdlist)
                    for folder in fdlist:
                        if len(os.listdir(folder)) == 0:
                            continue
                        self.imageAllFalse.append([[os.path.join(folder, i) for i in sorted(os.listdir(folder))], 1.])

        else:
            numPosi = 0
            numneg = 0
            self.image_name_positive  = []
            for folder in self.path_test:
                self.image_name_positive += [os.path.join(folder,i) for i in os.listdir(folder)]
            for steps,fd in enumerate(self.image_name_positive):
                fdlist =[]
                findDepestFolder(fd,fdlist)
                for folder in fdlist:
                    if len(os.listdir(folder)) == 0:
                        continue
                    if '1_' in fd:
                        label = 1
                    else:
                        label = 0
                    self.imageAllTrue.append([[os.path.join(folder,i) for i in sorted(os.listdir(folder))],label])
                    numPosi+=len(os.listdir(folder))

        self.train = []
        self.val = []
        self.imageAll = self.imageAllTrue
        # for i in range(len(self.imageAll)):
        #     if i%4 != 0:
        #         if self.imageAll[i][1] == 1.:
        #             self.train.append([self.imageAll[i]])
        #         else:
        #             for j in range(4):
        #                 self.train.append([self.imageAll[i],'train'])
        #     else:
        #         if self.imageAll[i][1] == 1.:
        #             self.train.append([self.imageAll[i],'val'])
        #         else:
        #             for j in range(4):
        #                 self.train.append([self.imageAll[i], 'val'])
        # print('\033[1;31m nums of posi is {}, num of neg is {} \033[0m'.format(numPosi , numneg,))

        if phase == 'train':
            # self.path = '/data2/gaozebin/ProstateX/data/aug_train'
            self.rotate = True
            # self.image_name = [os.path.join(self.path, i) for i in sorted(self.image_name)]
            self.imageList = self.imageAll
            if policy is not None:
                pass
        elif phase == 'val':
            self.rotate = True
            #self.image_name = [os.path.join(self.path, i) for i in os.listdir(self.path)]
            self.imageList = self.imageAll
            if policy is not None:
                pass
            # self.path = '/data2/gaozebin/ProstateX/data/aug_val'

    def resize(self,image,name,needtrans = False):
        if needtrans == True:
            image = image.transpose((2,1,0))
        # print(image.shape)
        try:
            image= cv2.resize(image,self.wh, interpolation=cv2.INTER_CUBIC)
        except:
            print('cannot convert',name)
        if np.random.randint(0,101)>99:
            cv2.imwrite('./img.png',image)
        image =  image.transpose((2,1,0))
        return image

    def parse_age(self,ageinfo,parse_interval = 5,maxage = 100):
        age = ageinfo
        ageparse = np.zeros((maxage//parse_interval))
        ind = age//parse_interval
        ageparse[ind] = 1.
        return ageparse
    def Rotate(self,img):
        angle = random.random()*360
        scipy.ndimage.rotate(img, angle, axes=(1, 2), reshape=False, output=None, order=3, mode='constant', cval=0.0,
                             prefilter=True)
        if np.random.randint(0, 101) > 99:
            image = img.transpose((2, 1, 0))
            cv2.imwrite('./imgrotate.png', image)
        return img

    def __getitem__(self, item):
        foldername = self.imageList[item][0][0].split('/')[5]
        label= self.imageList[item][1]
        npath = os.path.join(self.savepathtest,str(label)+'_'+foldername+'_'+str(item))
        os.mkdir(npath)
        for i in range(len(self.imageList[item][0])):

            imgname = self.imageList[item][0][i].split('/')[-1]
            namepatient = self.imageList[item][0][i].split('/')[-2]
            if 'pos' not in namepatient or 'neg' not in namepatient:
                copyfile(self.imageList[item][0][i], os.path.join(npath,namepatient+imgname))
            else:
                copyfile(self.imageList[item][0][i], os.path.join(npath, imgname))
        return self.imageList[item]
    def __len__(self):
        return len(self.imageList)
def get_opts():
    parser = argparse.ArgumentParser(description='SEnet50')

    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-epochs', type=int, default=200)
    parser.add_argument('-r', type=int, default=3)
    parser.add_argument('-use_cuda', default=True)
    parser.add_argument('-print_every', type=int, default=40)
    opt = parser.parse_args()
    return opt
if __name__ == '__main__':
    db = DataBowl("train")
    dl = torch.utils.data.DataLoader(
        db,
        batch_size=1,
        shuffle=True,
        num_workers = 16,
        )
    lidc = 0
    y301 = 0
    for i,j in enumerate(dl):
        if i == 1:
            print(j[1][0])
            print(j[0][0][0])
    print(lidc,y301)