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
from natsort import natsorted
import scipy
from PIL import ImageOps, ImageEnhance, ImageFilter, Image
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

def randomColor(image,randoms):
    """
    颜色抖动
    """

    color_image = ImageEnhance.Color(image).enhance(randoms[0])  # 调整图像的饱和度

    brightness_image = ImageEnhance.Brightness(color_image).enhance(randoms[1])  # 调整图像的亮度

    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(randoms[2])  # 调整图像对比度

    contrast_image = contrast_image.filter(ImageFilter.GaussianBlur(radius=randoms[3]))
    return ImageEnhance.Sharpness(contrast_image).enhance(randoms[4])  # 调整图像锐度
def crop(img):
    img = np.pad(img,((20,20),(20,20),(0,0)),'constant',constant_values=0)
    rand1 = np.random.randint(0,20) / 100.
    rand2 = np.random.randint(80,100)/ 100.
    rand3 = np.random.randint(0,20) / 100.
    rand4 = np.random.randint(80, 100) / 100.
    w,h,c = img.shape
    img = img[int(w*rand1):int(w*rand2),int(h*rand3):int(h*rand4),:]

    #print(img.shape)
    return img
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
    num = np.random.randint(1,30)
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
def jumpGetNegtive(dirlist,total,b,numSeries):
    rd = np.random.randint(0, 100)
    if rd < 10:
        dirlist1 = [dirlist[len(dirlist) // 2]]
    elif rd < 30:
        total_num = np.random.randint(2, numSeries)
        dirlist1 = dirlist[(len(dirlist) // 2 - total_num // 2):len(dirlist) // 2 + total_num // 2]
    else:
        dirlist1 = dirlist
    total = len(dirlist1)
    if total > numSeries:
        b = total // numSeries
    else:
        b = 1
    return jumpGet(dirlist1, total, b, numSeries)
def jumpGet(dirlist,total,b,numSeries):
    retlist = []
    if total >b*numSeries:
        i = np.random.randint(0,total-b*numSeries)
    else:
        i = 0
    while i<total:
        if random.randint(1,100)>100:
            size = np.random.randint(1,20)
            retlist.append(addFlip(addNoise(dirlist[i],dirlist[i].shape[-1],size)))
        else:
            retlist.append(addFlip(dirlist[i]))
        i+=b
    #if len(retlist) == 0:
        #print(name,i,b,total,len(dirlist),len(retlist),retlist)
    if len(retlist) == 1:
        imgblock = retlist[0]
    else:
        if np.random.randint(0,20) > 4:
            random.shuffle(retlist)
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
            cv2.imwrite('./lookimages/'+str(i)+'.png',imgblock[i,:,:,:].transpose(2,1,0))
    return imgblock
class DataBowl(Dataset):
    def __init__(self, phase='train', policy = None,wh = (256,256
                                                          ),numSeries = 20,
                 path_train = '/data/chenxiangru/covidData/TrainCov3-17',
                 path_val = '/data/chenxiangru/covidData/TestCov3-17'):
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.policy = policy
        self.wh = wh
        self.numSeries =numSeries
        self.path_train =path_train
        self.path_val = path_val
        self.stdmean = {
            'mean1':0.1552138492848401,
            'mean2':0.2855536948533397,
            'mean3':0.4463296922469498,
            'meanB':0.618249474927012,
            'meanG': 0.6182388542992485,
            'meanR': 0.6182548740982545,
            'std1':0.1476093989184158,
            'std2':0.21078306810335878,
            'std3':0.2917685697937044,
            'stdB':0.35597394452209713,
            'stdG':0.35596497169516284,
            'stdR':0.3559756108664847,

        }
        meanmap1 = np.ones((1,1,wh[0],wh[1]))*self.stdmean['meanB']
        meanmap2 = np.ones((1,1,wh[0],wh[1]))*self.stdmean['meanG']
        meanmap3 = np.ones((1,1,wh[0],wh[1]))*self.stdmean['meanR']
        self.meanmap = np.concatenate([meanmap1,meanmap2,meanmap3],axis = 1)
        stdmap1 = np.ones((1,1, wh[0], wh[1])) * self.stdmean['stdB']
        stdmap2 = np.ones((1,1, wh[0], wh[1])) * self.stdmean['stdG']
        stdmap3 = np.ones((1,1, wh[0], wh[1])) * self.stdmean['stdR']
        self.stdmap = np.concatenate([stdmap1, stdmap2, stdmap3], axis=1)

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
            self.image_name_train = [os.path.join(self.path_train,i) for i in os.listdir(self.path_train)]
            self.image_name_val = [os.path.join(self.path_val, i) for i in os.listdir(self.path_val)]


        print('\033[1;31m nums of train is {}, num of val is {}\033[0m'.format(len(self.image_name_train) , len(self.image_name_val)))

        if phase == 'train':
            # self.path = '/data2/gaozebin/ProstateX/data/aug_train'
            self.rotate = True
            # self.image_name = [os.path.join(self.path, i) for i in sorted(self.image_name)]
            self.imageList = self.image_name_train
            if policy is not None:
                self.parse_policy(policy)
        elif phase == 'val':
            self.rotate = True
            #self.image_name = [os.path.join(self.path, i) for i in os.listdir(self.path)]
            self.imageList = self.image_name_val
            if policy is not None:
                self.parse_policy(policy)
            # self.path = '/data2/gaozebin/ProstateX/data/aug_val'

    def resize(self,image,name = None,needtrans = False):
        if needtrans == True:
            try:
                image = image.transpose((2,1,0))
            except:
                print('error in resize',name)

        # print(image.shape)
        try:
            image= cv2.resize(image,self.wh, interpolation=cv2.INTER_CUBIC)
        except:
            print('error in resize')
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

        images = []
        label = int(self.imageList[item].split('/')[-1].split(',')[-1].split('.')[0])
        fs = natsorted(os.listdir(self.imageList[item]))
        randomlist = [
            np.random.randint(0, 31) / 10. ,
            np.random.randint(5, 15) / 10., # 随机因子
            np.random.randint(5, 15) / 10.,  # 随机因1子
            np.random.randint(0,2),
            np.random.randint(0, 31) / 10.
            ]
        for i in range(len(fs)):
            #print(self.train[item][0][i])
            img = Image.open(os.path.join(self.imageList[item], fs[i])).convert("RGB")
            img  = randomColor(img,randomlist)

            img = np.array(img)
            # print(img.shape)
            # img = crop(img)
            # self.resize(img, self.imageList[item])
            img = Image.fromarray(img)
            # img = img.rotate(np.random.randint(0,360))
            img = img.resize(self.wh, Image.BILINEAR)
            img = np.array(img).transpose(2, 1, 0)
            # print(img.shape)
            images.append(np.expand_dims(img, axis=0))
        numSeries = self.numSeries
        length= len(fs)
        if length>numSeries:
            b = length // numSeries
        else:
            b=1
        if self.imageList[item][1] == 0.:
            if np.random.randint(0,10)<3:
                imgblock =jumpGetNegtive(images,length,b,numSeries)
            else:
                imgblock =jumpGet(images,length,b,numSeries)
        else:
            imgblock = jumpGet(images,length,b,numSeries)

        if self.policy is None:
            self.image = torch.Tensor((np.ascontiguousarray(imgblock/255.)))
            # self.image = torch.Tensor((np.ascontiguousarray((imgblock/255. - self.meanmap)/self.stdmap)))
            return self.image, torch.tensor(label).float()
        else:
            #self.image = self.ApplyPbaAug(copy.deepcopy(self.image))
            # self.image = torch.Tensor(np.ascontiguousarray((imgblock/255. - self.meanmap)/self.stdmap))
            self.image = torch.Tensor(np.ascontiguousarray(imgblock/255.))
            # if np.random.randint(0, 101) > 99:
            #     image1 = self.image.numpy().transpose(1,2,0)*255.
            #     cv2.imwrite('./imgAg.png', image1)
        #images = [torch.Tensor(images[im]) for im in range(len(images))]
            return self.image  ,torch.tensor(label).float()

    def __len__(self):

        return len(self.imageList)


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

