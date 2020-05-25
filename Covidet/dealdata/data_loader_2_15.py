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
import augmentation_transforms_hp as augmentation_transforms

import scipy

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
        self.path_positive = '/sharedata/gaozebin/FightNCP/positive'
        self.path_negtive = '/sharedata/gaozebin/FightNCP/negtive_normal'
        self.testpositive = '/sharedata/gaozebin/FightNCP/positive1'
        self.testnegtive = '/sharedata/gaozebin/FightNCP/negtive1'
        self.stdmean = {
            'mean1':0.1552138492848401,
            'mean2':0.2855536948533397,
            'mean3':0.4463296922469498,
            'std1':0.1476093989184158,
            'std2':0.21078306810335878,
            'std3':0.2917685697937044,
        }
        self.stdmeanAddNoise={'mean1': 0.4951994540168879, 'mean2': 0.49413567928584506, 'mean3': 0.4905329068393595, 'std1': 0.08913075419725634, 'std2': 0.12143960210148051, 'std3': 0.11874566770061311}

        self.imageAllTrue = []
        self.imageAllFalse = []
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
            for f in self.test_negtive:
                self.image_name = [os.path.join(self.test_negtive, i) for i in os.listdir(self.path_positive)]
                for fd in self.image_name_positive:
                    fdlist = []
                    findDepestFolder(fd, fdlist)
                    for folder in fdlist:
                        if len(os.listdir(folder)) == 0:
                            continue
                        self.imageAllFalse.append([[os.path.join(folder, i) for i in sorted(os.listdir(folder))], 1.])

        else:
            self.image_name_positive = [os.path.join(self.path_positive,i) for i in os.listdir(self.path_positive)]
            for fd in self.image_name_positive:
                fdlist =[]
                findDepestFolder(fd,fdlist)
                for folder in fdlist:
                    if len(os.listdir(folder)) == 0:
                        continue
                    self.imageAllTrue.append([[os.path.join(folder,i) for i in sorted(os.listdir(folder))],1.])
            self.image_name_neg = [os.path.join(self.path_negtive, i) for i in os.listdir(self.path_negtive)]
            for fd in self.image_name_neg:
                fdlist = []
                findDepestFolder(fd, fdlist)
                for folder in fdlist:
                    self.imageAllFalse.append([[os.path.join(folder, i) for i in sorted(os.listdir(folder))], 0.])

        self.train = []
        self.val = []
        self.imageAll = self.imageAllFalse*4+self.imageAllTrue
        random.shuffle(self.imageAll)
        for i in range(len(self.imageAll)):
            if i%4 != 0:
                self.train.append(self.imageAll[i])
            else:
                self.val.append(self.imageAll[i])
        print('\033[1;31m nums of train is {}, num of val is {}\033[0m'.format(len(self.train) , len(self.val)))

        if phase == 'train':
            # self.path = '/data2/gaozebin/ProstateX/data/aug_train'
            self.rotate = True
            # self.image_name = [os.path.join(self.path, i) for i in sorted(self.image_name)]
            self.imageList = self.train
            if policy is not None:
                self.parse_policy(policy)
        elif phase == 'val':
            self.rotate = True
            #self.image_name = [os.path.join(self.path, i) for i in os.listdir(self.path)]
            self.imageList = self.val
            if policy is not None:
                self.parse_policy(policy)
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
    def parse_policy(self, policy_emb, augmentation_transforms=augmentation_transforms):
        self.policy = []
        #print(policy_emb)
        num_xform = augmentation_transforms.NUM_HP_TRANSFORM
        xform_names = augmentation_transforms.HP_TRANSFORM_NAMES
        assert len(policy_emb
                   ) == 2 * num_xform, 'policy was: {}, supposed to be: {}'.format(
            len(policy_emb), 2 * num_xform)
        for i, xform in enumerate(xform_names):
            self.policy.append((xform, policy_emb[2 * i] / 10., policy_emb[2 * i + 1]))
        return
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
        if self.phase == 'test':
            self.image = np.load(self.test[item]).astype(np.uint8)
            self.image = self.image[::-1,:,:]
            #self.image = data_process(self.image)
            name = self.test[item].split('/')[-1].split('_')
            namestr = [name[0]+name[1]]
            self.image = self.resize(self.image)
            self.image = torch.Tensor((np.ascontiguousarray(self.image/255.)))
            return self.image, namestr,[]
        else:
            images = []
            for i in range(len(self.imageList[item][0])):
                #print(self.train[item][0][i])
                img = cv2.imread(self.imageList[item][0][i])
                images.append(np.expand_dims(self.resize(img,self.imageList[item][0][i]),axis=0))
            numSeries = self.numSeries
            length= len(self.imageList[item][0])
            if length>numSeries:
                b = length // numSeries
            else:
                b=1
            if self.imageList[item][1] == 0.:
                if np.random.randint(0,10)<3:
                    imgblock =jumpGetNegtive(images,length,b,self.imageList[item][0],numSeries)
                else:
                    imgblock =jumpGet(images,length,b,self.imageList[item][0],numSeries)
            else:
                imgblock = jumpGet(images,length,b,self.imageList[item][0],numSeries)

            self.label = self.imageList[item][1]
        if self.policy is None:
            self.image = torch.Tensor((np.ascontiguousarray(imgblock/255.)))
            return self.image, torch.tensor(self.label).float()
        else:
            #self.image = self.ApplyPbaAug(copy.deepcopy(self.image))
            self.image = torch.Tensor(np.ascontiguousarray(imgblock/255.))
            # if np.random.randint(0, 101) > 99:
            #     image1 = self.image.numpy().transpose(1,2,0)*255.
            #     cv2.imwrite('./imgAg.png', image1)
        #images = [torch.Tensor(images[im]) for im in range(len(images))]
            return self.image  ,torch.tensor(self.label).float()

    def __len__(self):
        if self.phase == 'test':
            return len(self.test)
        if self.phase =='val':
            return len(self.val)
        if self.phase =='train':
            return len(self.train)
        return 0

    def ApplyPbaAug(self,images):
        return augmentation_transforms.apply_policy(self.policy, images ,'eye','eye', image_size=512)

def get_dataloader(config):
    # MNIST Dataset
    train_set = DataBowl(phase = 'train',policy=config['hp_policy'])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                                               drop_last=True)

    val_set = DataBowl(phase='val')
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                                             drop_last=False)

    # Data Loader (Input Pipeline)

    return train_loader, val_loader
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
        )
    for i,(x,y) in enumerate(dl):
        if i >1:
            break
        print(x.size())

