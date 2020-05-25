from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
import pandas as pd
import pydicom as dicom

from torch.utils.data import Dataset
from torchvision import transforms
try:
    from dataloader import custom_trans as tr
except:
    import custom_trans as tr


FPATH = '/data/gaozb/seg_Pneumonia/'
LMAP = {0: 0, 1: 1, 2: 0}

np.random.seed(2333)


class DataBowl(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 3

    def __init__(self,
                 args=None,
                 base_dir=FPATH,
                 phase='train',
                 split=0.2,
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        
        super().__init__()
        self.phase = phase
        self._base_dir = base_dir
        if phase == 'train':
            #self.csv = pd.read_csv('/data/gaozb/seg_Pneumonia/data/csv/train_Y_nocrop.csv')
            self.csv = pd.read_csv('/data/gaozb/seg_Pneumonia/data/csv/train_lobe.csv')
        elif phase == 'val':
            #self.csv = pd.read_csv('/data/gaozb/seg_Pneumonia/data/csv/val_Y_nocrop.csv')
            self.csv = pd.read_csv('/data/gaozb/seg_Pneumonia/data/csv/val_lobe.csv')
        elif phase == 'test':
            self.csv = pd.read_csv('/data/gaozb/seg_Pneumonia/data/csv/test_Y52.csv')
        elif phase == 'inference':
            self.csv = []
            self.total = {}
            # for root, dirs, files in os.walk('/data/gaozb/CYY'):
            #     for f in files:
            #         if f != 'DIRFILE':
            #             self.csv.append(os.path.join(root, f))
            #for root, dirs, files in os.walk('/data/gaozb/seg_Pneumonia/50-50_deal'):
            for root, dirs, files in os.walk('/data/gaozb/CYY_png/15.107/DICOM/'):
                for f in files:
                    if f.endswith(('jpg', 'png', 'jpeg', 'bmp', 'JPG', 'PNG', 'JPEG', 'BMP')):
                        self.total[os.path.join(root, f)] = os.path.join(root, f)
            self.done = {}
            #for root, dirs, files in os.walk('/data/gaozb/seg_Pneumonia/50-50_deal'):
            #    for f in files:
            #        if f.endswith(('jpg', 'png', 'jpeg', 'bmp', 'JPG', 'PNG', 'JPEG', 'BMP')):
            #            self.done[os.path.join(root, f).replace('/data/gaozb/seg_Pneumonia/50-50_deal', '/data/gaozb/CYY_png')] = os.path.join(root, f)

            tmp = set(self.total.keys()).difference(set(self.done.keys()))
            for k in tmp:
                self.csv.append(self.total[k])
        self.index_list = list(range(len(self.csv)))
        np.random.shuffle(self.index_list)
        
        #if self.phase == 'train':
        #    self.index_list = self.index_list[int(len(self.index_list) * 0.2):]
        #elif self.phase == 'val':
        #    self.index_list = self.index_list[:int(len(self.index_list) * 0.2)]

        self.args = args

        print('Number of images in {}: {:d}'.format(phase, len(self.index_list)))

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        index = self.index_list[index]
        if self.phase == "inference":
            name = self.csv[self.index_list[index]]
            if name.endswith(('jpg', 'png', 'jpeg', 'bmp', 'JPG', 'PNG', 'JPEG', 'BMP')):
                _img = Image.open(self.csv[self.index_list[index]]).convert('RGB')
            else:
                _img = Image.fromarray(self.read_dcm(name)).convert('RGB')
            sample = {'image': self.transform_test(_img), 'name': name}
            return sample

        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}
        name = os.path.join(self._base_dir, self.csv['image'][index])
        # print(name)
        if self.phase == "train":
            return self.transform_tr(sample)
        elif self.phase == 'val':
            sample = self.transform_val(sample)
            return {**sample, **{"name":name}}
        elif self.phase == 'test':
            sample = self.transform_val(sample)
            return {**sample, **{"name":name}}
        
    def _make_img_gt_point_pair(self, index):
        _img = os.path.join(self._base_dir, self.csv['image'][index])
        _target = os.path.join(self._base_dir, self.csv['label'][index])

        _img = Image.open(_img).convert('RGB')
        _target = Image.open(_target)
        # print(self.csv['image'][index], _img.size, _target.size)
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomCutBlock(value=255),
            tr.RandomGaussianBlur(),
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(512, 512),
            tr.RandomRotate(degree=40),
            tr.FixedResize(self.args['resize']),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixedResize(self.args['resize']),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_test(self, img):
        img = img.resize((self.args['resize'], self.args['resize']), Image.BILINEAR)
        img = np.array(img).astype(np.float32)
        img = img / 255.
        img -= (0.485, 0.456, 0.406)
        img /= (0.229, 0.224, 0.225)
        return img.transpose(2, 0, 1)


    def ww_wc(self, img,k='309'):
        # 调整窗位窗宽
        ref_dict = {"L":[-600,1600],"LUNA":[-500,1500],"309":[-650,1200]}
        
        wcenter = ref_dict[k][0]
        wwidth = ref_dict[k][1]
        minvalue = (2 * wcenter - wwidth) / 2.0 + 0.5
        maxvalue = (2 * wcenter + wwidth) / 2.0 + 0.5
        
        dfactor = 255.0 / (maxvalue - minvalue)
        
        zo = np.ones(img.shape) * minvalue
        Two55 = np.ones(img.shape) * maxvalue
        img = np.where(img < minvalue, zo, img)
        img = np.where(img > maxvalue, Two55, img)
        img = ((img - minvalue) * dfactor).astype('uint8')
        
        return img
    
    def read_dcm(self, path):
    
        slices = dicom.read_file(path)

        image = slices.pixel_array.astype(np.int16)
        #assert image.shape == (512, 512)
        # 设置边界外的元素为0
        image[image == -2000] = 0

        intercept = slices.RescaleIntercept
        slope = slices.RescaleSlope
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)

        image += np.int16(intercept)
        image = self.ww_wc(image)
        
        return image

    
    def __str__(self):
        return 'CAMELYON17 (split=' + str(self.split) + ')'


if __name__ == '__main__':
    from utils import decode_segmap
    from torch.utils.data import DataLoader
    from PIL import Image
    import pdb

    voc_train = DataBowl(args={"resize":224 })

    dataloader = DataLoader(voc_train, batch_size=128, shuffle=True, num_workers=16)

    for ii, sample in enumerate(dataloader):
        print(sample['label'].numpy().min(), sample['label'].numpy().max())
#        if sample["label"].numpy().max() <= 0:
#            continue
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj])
            segmap = 127 * tmp
            segmap = segmap.astype(np.uint8)
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            img = Image.fromarray(img_tmp)
            segmap = Image.fromarray(segmap)
            img.save('tmp/img%s.jpg' %str(jj))
            segmap.save('tmp/gt%s.png' %str(jj))

