import os
import pdb
import cv2
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch

from dataloader.dataloader import DataBowl
from util import Evaluator
from myconfig import convert_to_params, create_parser
from models.deeplab import DeepLab


def dice_coeff(pred, target):

    pred = np.dstack(pred)
    target = np.dstack(target)
    pred = pred.reshape(-1)
    target = pred.reshape(-1)
    smooth = 1.
    intersection = np.sum(pred * target)
 
    return (2. * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)

def eval(model, model_path, dl, phase='test'):
    model.cuda()
    try:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path)["weight"])
    except:
        model.load_state_dict(torch.load(model_path)["weight"])
    model.eval()

    pre_cls = []
    pre_prob = []
    
    if phase == 'val':
        score = Evaluator()
        score.reset()

    for i, sample in enumerate(tqdm(dl)):
        x = sample['image'].cuda()
        name = sample['name']
        
        with torch.no_grad():
            logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).detach().cpu().numpy().astype(np.uint8)
        probs = probs.detach().cpu().numpy()[:, 1, ...]
        if phase == 'val':
            score.add_batch(y, preds)
        x = x.detach().cpu().numpy().transpose([0, 2, 3, 1])
        x *= (0.229, 0.224, 0.225)
        x += (0.485, 0.456, 0.4056)
        x *= 255
        x = x.astype(np.uint8)
        preds = preds * 255

        # pdb.set_trace()
        
        for one in range(len(preds)):
            pre_cls.append(preds[one])
            pre_prob.append(probs[one])
            pic = np.hstack([
                x[one],
                cv2.cvtColor(preds[one], cv2.COLOR_GRAY2RGB)
            ])
            path = name[one].replace('CYY_png', 'CYY_pre1')
            dir_ = os.path.join('/', *path.split('/')[:-1])
            if not os.path.exists(dir_):
                os.makedirs(dir_)
            Image.fromarray(pic).save(path)
    return


if __name__ == "__main__":
    flags = create_parser()
    hparams = convert_to_params(flags)

    os.environ['CUDA_VISIBLE_DEVICES'] =hparams['gpu_num']

    ds = DataBowl(hparams, phase='inference')
    dl = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=16, shuffle=False)

    model = DeepLab()
    
    # model for seg pneu
    model_path = ['./checkpoints/model_pneu.pkl',]

    # model for seg lobe
    model_path = ['./checkpoints/model_lobe.pkl',]
    pre_total = []
    tta = 1
    for i, mp in enumerate(model_path):
        print("model path is {}.".format(mp))
        for j in range(tta):
            eval(model, mp, dl)
    print("Done")
    
