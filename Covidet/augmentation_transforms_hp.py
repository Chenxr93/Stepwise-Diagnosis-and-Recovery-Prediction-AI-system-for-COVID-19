# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Transforms used in the PBA Augmentation Policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import inspect
import random
import torch
import torch.nn
import torch.functional as F

import numpy as np
from PIL import ImageOps, ImageEnhance, ImageFilter, Image  # pylint:disable=g-multiple-import

from augmentation_transforms import random_flip, zero_pad_and_crop, cutout_numpy  # pylint: disable=unused-import
from augmentation_transforms import TransformFunction
from augmentation_transforms import ALL_TRANSFORMS, NAME_TO_TRANSFORM, TRANSFORM_NAMES,NAME_TO_TRANSFORM  # pylint: disable=unused-import
from augmentation_transforms import pil_wrap, pil_unwrap  # pylint: disable=unused-import
from augmentation_transforms import MEANS, STDS, PARAMETER_MAX  # pylint: disable=unused-import
from augmentation_transforms import _rotate_impl, _posterize_impl, _shear_x_impl, _shear_y_impl, _translate_x_impl, _translate_y_impl, _crop_impl, _solarize_impl, _cutout_pil_impl, _enhancer_impl
from scipy.ndimage.interpolation import rotate as scirotate
def apply_policy3D(policy, img, aug_policy, image_size, verbose=False):
    """Apply the `policy` to the numpy `img`.

  Args:
    policy: A list of tuples with the form (name, probability, level) where
      `name` is the name of the augmentation operation to apply, `probability`
      is the probability of applying the operation and `level` is what strength
      the operation to apply.
    img: Numpy image that will have `policy` applied to it.
    aug_policy: Augmentation policy to use.
    dset: Dataset, one of the keys of MEANS or STDS.
    image_size: Width and height of image.
    verbose: Whether to print applied augmentations.

  Returns:
    The result of applying `policy` to `img`.
  """
    if aug_policy == 'cifar10':
        count = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.5, 0.0])
    elif aug_policy == '3D':
        count = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.5, 0.0])
        #raise ValueError('Unknown aug policy.')
    else:
        count = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.5, 0.0])
    if count != 0:
        pil_img = img
        policy = copy.copy(policy)
        random.shuffle(policy)
        for xform in policy:
            assert len(xform) == 3
            name, probability, level = xform
            assert 0. <= probability <= 1.
            assert 0 <= level <= PARAMETER_MAX
            xform_fn = NAME_TO_TRANSFORM1[name].pil_transformer(
                probability, level, image_size)
            pil_img, res = xform_fn(pil_img)
            if verbose and res:
                print("Op: {}, Magnitude: {}, Prob: {}".format(name, level, probability))
            count -= res
            assert count >= 0
            if count == 0:
                break
        return pil_img
    else:
        return img

def apply_policy(policy, img, aug_policy, dset, image_size, verbose=False):
    """Apply the `policy` to the numpy `img`.

  Args:
    policy: A list of tuples with the form (name, probability, level) where
      `name` is the name of the augmentation operation to apply, `probability`
      is the probability of applying the operation and `level` is what strength
      the operation to apply.
    img: Numpy image that will have `policy` applied to it.
    aug_policy: Augmentation policy to use.
    dset: Dataset, one of the keys of MEANS or STDS.
    image_size: Width and height of image.
    verbose: Whether to print applied augmentations.

  Returns:
    The result of applying `policy` to `img`.
  """
    count = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.5, 0.0])
    if count != 0:
        pil_img = pil_wrap(img, dset)
        policy = copy.copy(policy)
        random.shuffle(policy)
        for xform in policy:
            assert len(xform) == 3
            name, probability, level = xform
            assert 0. <= probability <= 1.
            assert 0 <= level <= PARAMETER_MAX
            xform_fn = NAME_TO_TRANSFORM[name].pil_transformer(
                probability, level, image_size)
            pil_img, res = xform_fn(pil_img)
            if verbose and res:
                print("Op: {}, Magnitude: {}, Prob: {}".format(name, level, probability))
            count -= res
            assert count >= 0
            if count == 0:
                break
        return pil_unwrap(pil_img, dset, image_size)
    else:
        # pil_img = pil_wrap(img, dset)
        # return pil_unwrap(pil_img, dset, image_size)
        return img/255.


class TransformT(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, name, xform_fn):
        self.name = name
        self.xform = xform_fn

    def pil_transformer(self, probability, level, image_size):
        """Builds augmentation function which returns resulting image and whether augmentation was applied."""

        def return_function(im):
            res = False
            if random.random() < probability:
                if 'image_size' in inspect.getargspec(self.xform).args:
                    im = self.xform(im, level, image_size)
                else:
                    im = self.xform(im, level)
                res = True
            return im, res

        name = self.name + '({:.1f},{})'.format(probability, level)
        return TransformFunction(return_function, name)

    def str(self):
        return self.name
def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
    return int(level * maxval / PARAMETER_MAX)

################## Transform Functions ##################
identity = TransformT('identity', lambda pil_img, level: pil_img)
flip_lr = TransformT(
    'FlipLR', lambda pil_img, level: pil_img.transpose(Image.FLIP_LEFT_RIGHT))
flip_ud = TransformT(
    'FlipUD', lambda pil_img, level: pil_img.transpose(Image.FLIP_TOP_BOTTOM))
# pylint:disable=g-long-lambda
auto_contrast = TransformT(
    'AutoContrast',
    lambda pil_img, level: ImageOps.autocontrast(pil_img.convert('RGB')).convert('RGBA')
)
equalize = TransformT(
    'Equalize',
    lambda pil_img, level: ImageOps.equalize(pil_img.convert('RGB')).convert('RGBA')
)
invert = TransformT(
    'Invert',
    lambda pil_img, level: ImageOps.invert(pil_img.convert('RGB')).convert('RGBA')
)
# pylint:enable=g-long-lambda
blur = TransformT('Blur',
                  lambda pil_img, level: pil_img.filter(ImageFilter.BLUR))
smooth = TransformT('Smooth',
                    lambda pil_img, level: pil_img.filter(ImageFilter.SMOOTH))
rotate = TransformT('Rotate', _rotate_impl)
posterize = TransformT('Posterize', _posterize_impl)
shear_x = TransformT('ShearX', _shear_x_impl)
shear_y = TransformT('ShearY', _shear_y_impl)
translate_x = TransformT('TranslateX', _translate_x_impl)
translate_y = TransformT('TranslateY', _translate_y_impl)
crop_bilinear = TransformT('CropBilinear', _crop_impl)
solarize = TransformT('Solarize', _solarize_impl)
cutout = TransformT('Cutout', _cutout_pil_impl)
color = TransformT('Color', _enhancer_impl(ImageEnhance.Color))
contrast = TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))
brightness = TransformT('Brightness', _enhancer_impl(ImageEnhance.Brightness))
sharpness = TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))

HP_TRANSFORMS = [
    auto_contrast,
    equalize,
    invert,
    rotate,
    posterize,
    solarize,
    color,
    contrast,
    brightness,
    sharpness,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
    cutout,
]

NAME_TO_TRANSFORM = {t.name: t for t in HP_TRANSFORMS}
HP_TRANSFORM_NAMES = NAME_TO_TRANSFORM.keys()
NUM_HP_TRANSFORM = len(HP_TRANSFORM_NAMES)


def flip3D1(block,level):
    return block[::-1,:,:]
def flip3D2(block,level):
    return block[:,::-1,:]
def flip3D3(block,level):
    return block[:,:,::-1]
def translation3D1(block,level):
    block = np.pad(block, pad_width=((3, 3), (3, 3), (3, 3)), mode = 'minimum')
    step = int_parameter(level, 3)
    if random.random() > 0.5:
        step = -step
        step += 3
    return block[step:32+step,3:35,3:15]
def translation3D2(block,level):
    block = np.pad(block, pad_width=((3, 3), (3, 3), (3, 3)), mode = 'minimum')
    step = int_parameter(level, 3)
    if random.random() > 0.5:
        step = -step
        step+=3
    return block[3:35, step:32+step,3:15]
def translation3D3(block,level):
    block = np.pad(block, pad_width=((3, 3), (3, 3), (3, 3)), mode = 'minimum')
    step = int_parameter(level, 2)
    if random.random() > 0.5:
        step = -step
        step+=2
    return block[3:35, 3:35,step:step+12]
def rotate3D1(block,level):
    angle = int_parameter(level, 180)*1.+np.random.rand()*10
    order = 3
    if random.random() > 0.5:
        block = scirotate(block, angle, axes=(1, 2), reshape=False, order=order, mode='constant', cval=0.0,
                       prefilter=False)
    return block
def rotate3D2(block,level):
    angle = int_parameter(level, 180)*1.+np.random.rand()*10
    order = 3
    if random.random() > 0.5:
        block = scirotate(block, angle, axes=(0, 2), reshape=False, order=order, mode='constant', cval=0.0,
                       prefilter=False)
    return block
def rotate3D3(block,level):
    angle = int_parameter(level, 180)*1.+np.random.rand()*10
    order = 3
    if random.random() > 0.5:
        block = scirotate(block, angle, axes=(0, 1), reshape=False, order=order, mode='constant', cval=0.0,
                       prefilter=False)
    return block

Flip3D1 = TransformT('Flip3D1', flip3D1)
Flip3D2 = TransformT('Flip3D2', flip3D2)
Flip3D3 = TransformT('Flip3D3', flip3D3)
Translation3D1 = TransformT('Translation3D1', translation3D1)
Translation3D2 = TransformT('Translation3D2', translation3D2)
Translation3D3 = TransformT('Translation3D3', translation3D3)
Rotate3D1 = TransformT('Rotate3D1', rotate3D1)
Rotate3D2 = TransformT('Rotate3D2', rotate3D2)
Rotate3D3 = TransformT('Rotate3D3', rotate3D3)

HP_TRANSFORM_NAMES1 = [Flip3D1,Flip3D2,Flip3D3,
                       Translation3D1,
                       Translation3D2,
                       Translation3D3,
                       Rotate3D1,
                       Rotate3D2,
                       Rotate3D3,
                       ]

NAME_TO_TRANSFORM1 = {t.name: t for t in HP_TRANSFORM_NAMES1}
HP_TRANSFORM_NAMES1 = NAME_TO_TRANSFORM1.keys()
NUM_HP_TRANSFORM1 = len(HP_TRANSFORM_NAMES1)
