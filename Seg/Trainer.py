import os
import time
import math
import copy
import pdb
import json

import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn import DataParallel

from dataloader.dataloader import  DataBowl
from unet.unet_model import UNet
from models.deeplab import DeepLab
from myconfig import create_parser, convert_to_params
from util import Evaluator
from dataloader.utils import decode_segmap
from sync_batchnorm import convert_model


def dice_coeff(pred, target):

    pred = np.dstack(pred)
    target = np.dstack(target)
    pred = pred.reshape(-1)
    target = pred.reshape(-1)
    smooth = 1.
    intersection = np.sum(pred * target)
 
    return (2. * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)


class Trainer(nn.Module):

    def __init__(self, hparams):
        super(Trainer, self).__init__()
        self.hparams = hparams
        self.check_dir = hparams['check_dir']
        self.writer = SummaryWriter(logdir=self.check_dir)

        self.lr = hparams['init_lr']
        self.best_models = {}
        self.evaluator = Evaluator(num_class=3)
        np.random.seed(0)

        self.data_set = DataBowl(args=self.hparams, phase='train')
        self.data_loader = torch.utils.data.DataLoader(self.data_set, shuffle=True,
                batch_size=self.hparams['tr_bs'],
                num_workers=16)

        data_set_val = DataBowl(args=self.hparams, phase='val')
        self.data_loader_val = torch.utils.data.DataLoader(data_set_val, shuffle=True,
                batch_size=self.hparams['tr_bs'],
                num_workers=16)

        self.tr_batch_nums = len(self.data_loader)
        self.total_tr_batch_nums = self.tr_batch_nums * hparams['epochs']
        self.model = DeepLab()
        #self.model = UNet(3, 2)
        self.model = self.model.cuda() #convert_model(self.model).cuda()
        self.model = DataParallel(self.model)
        # self.model.load_state_dict(torch.load('./checkpoints/experiment4/model_0.7474540262426985.pkl')['weight'])
        # self.criterion = nn.BCELoss()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        param_to_update = self.model.parameters()
        self.optimizer = torch.optim.Adam(param_to_update, lr=self.lr, weight_decay=1e-6)

    def _val_loop(self, epoch):
        labelT = []
        pre_cls = []
        pre_prob = []
        
        self.model.eval()
        total_loss = 0
        self.evaluator.reset()
        for i, sample in enumerate(tqdm(self.data_loader_val)):
            x = sample['image'].cuda()
            y = sample['label'].long().cuda()

            with torch.no_grad():
                seg = self.model(x)
            
            probs = torch.softmax(seg, dim=1)
            preds = torch.argmax(probs, dim=1)

            loss = self.criterion(seg, y)

            total_loss += loss.item()

            y = y.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            probs = probs.detach().cpu().numpy()[:, 1, ...]        
            self.evaluator.add_batch(y, preds)

            for one in range(y.shape[0]):
                labelT.append(y[one])
                pre_cls.append(preds[one])
                pre_prob.append(probs[one])
            
            if (i+1) % 50 == 0 or i == len(self.data_loader_val) - 1:
                dice = dice_coeff(pre_prob, labelT)
                mIoU, acc, fwIoU = self.evaluator.get_result()
                tqdm.write("=====================Iter: {}===================".format(i+1))
                tqdm.write("Loss:\t{:.2}\nmIoU:\t{:.4}\nfwIoU:\t{:.2}\nAcc:\t{:.2}\nDice:\t{:.4}".format(
                    loss.item(),
                    mIoU, fwIoU, acc, dice))
                tqdm.write("================================================")
        self.writer.add_scalar("val/loss", total_loss, epoch)
        self.writer.add_scalar("val/mIoU", mIoU, epoch)
        self.writer.add_scalar("val/dice", dice, epoch)        
        self.writer.add_scalar("val/acc", acc, epoch)
        self.best_models["{}".format(mIoU)] = {"weight": copy.deepcopy(self.model.state_dict()),
                                                        "dice":dice,
                                                        "mIoU":mIoU }
        self.keep_models()
        self.show_img(x, y, preds, epoch, 'val')
        return 0

    def _train_loop(self, epoch):
        """Trains the model `m` for one epoch."""
        start_time = time.time()
        labelT = []
        pre_cls = []
        pre_prob = []
        total_loss = 0
        self.model.train()
        self.evaluator.reset()
        for i, sample in enumerate(tqdm(self.data_loader)):
            x = sample['image'].cuda()
            y = sample['label'].long().cuda()

            lr_to_log = self.adjust_lr(i, epoch)
            self.writer.add_scalar("train/lr", lr_to_log, epoch)
            self.optimizer.zero_grad()

            seg = self.model(x)
            loss = self.criterion(seg, y)

            total_loss += loss

            loss.backward()
            self.optimizer.step()
            
            probs = torch.softmax(seg, dim=1)
            preds = torch.argmax(probs, dim=1)

            y = y.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            probs = probs.detach().cpu().numpy()[:, 1, ...]            
            self.evaluator.add_batch(y, preds)

            for one in range(y.shape[0]):
                labelT.append(y[one])
                pre_cls.append(preds[one])
                pre_prob.append(probs[one])

            if (i+1) % 100 == 0 or i == len(self.data_loader) - 1:
                dice = dice_coeff(pre_prob, labelT)
                mIoU, acc, fwIoU = self.evaluator.get_result()
                tqdm.write("=====================Iter: {}===================".format(i+1))
                tqdm.write("Loss:\t{:.2}\nmIoU:\t{:.2}\nfwIoU:\t{:.2}\nAcc:\t{:.2}\nDice:\t{:.4}\n".format(
                    loss.item(),
                    mIoU, fwIoU, acc, dice))
                tqdm.write("================================================")
                self.show_img(x, y, preds, epoch*len(self.data_loader)+i)
        self.writer.add_scalar("train/loss", total_loss, epoch)
        self.writer.add_scalar("train/mIoU", mIoU, epoch)
        self.writer.add_scalar("train/dice", dice, epoch)
        self.writer.add_scalar("train/fwIoU", fwIoU, epoch)
        self.writer.add_scalar("train/acc", acc, epoch)

        self.show_img(x, y, preds, epoch)
        return 0

    def run_model(self, epoch):
        """Trains and evalutes the image model."""
        training_accuracy = self._train_loop(epoch)
        tqdm.write("\033[1;34m One epoch trained and acc is{}.\033[0m".format(training_accuracy))
        valid_accuracy = self._val_loop(epoch)
        tqdm.write("\033[1;35m One epoch valed and acc is{}.\033[0m".format(valid_accuracy))
        return training_accuracy, valid_accuracy

    def show_img(self, x, y, preds, epoch, phase='train'):

        gt = np.expand_dims(y, 1)
        gt = gt.astype(np.uint8)
        #tqdm.write('gt' + str(gt.max()) + '_' + str(np.sum(gt)))
        pr = np.expand_dims(preds, 1)
        pr = pr.astype(np.uint8)
        #tqdm.write('pr:' + str(pr.max()) + '_' + str(np.sum(pr)))

        im = x.detach().cpu().numpy().transpose([0, 2, 3, 1])
        im *= (0.229, 0.224, 0.225)
        im += (0.485, 0.456, 0.406)
        im *= 255
        im = im.astype(np.uint8).transpose([0, 3, 1, 2])[:,::-1,...]
        self.writer.add_images("{}/gt".format(phase), gt, epoch)
        self.writer.add_images("{}/pr".format(phase), pr, epoch)
        self.writer.add_images("{}/img".format(phase), im, epoch)


    def adjust_lr(self, step, epoch, gamma=2.0):
        if epoch < 3:
            lr = self.lr * (epoch + 1) / 3
        else:
            current_step = epoch * self.tr_batch_nums + step
            current_step = current_step - self.tr_batch_nums * 3
            lr = self.lr * (1 + math.cos(math.pi * current_step / self.total_tr_batch_nums))/gamma
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr
        return lr

    def to_sunmmary_writer(self, phase, mIoU, kappa, acc, epoch):
        self.writer.add_scalar("{}/f1".format(phase), mIoU, epoch)
        self.writer.add_scalar("{}/kappa".format(phase), kappa, epoch)
        self.writer.add_scalar("{}/avg".format(phase), acc, epoch)
        self.writer.add_scalar("{}/loss".format(phase), loss, epoch)

    def keep_models(self):
        if len(self.best_models) <= 1:
            return
        else:
            key_list = sorted(list(self.best_models.keys()))
            for k in key_list[:-2]:
                del self.best_models[k]
        return

    def save_best_models(self):
        for k in self.best_models:
            fp = os.path.join(self.check_dir, "model_{}.pkl".format(k))
            torch.save(self.best_models[k], fp)
            tqdm.write("Save model to {}".format(fp))
        with open(os.path.join(self.check_dir, "hp.json"), 'w') as f:
            hp_json = json.dump(self.hparams, f)

    def close_writer(self):
        self.writer.export_scalars_to_json('./log.json')
        self.writer.close()


def main():
    flags = create_parser()
    hparams_config = convert_to_params(flags)
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_num
    trainer = Trainer(hparams_config)

    for i in trange(hparams_config['epochs']):
        train_acc, val_acc = trainer.run_model(i)
    trainer.save_best_models()
    trainer.close_writer()
    del(trainer)
    return

if __name__ == "__main__":
    main()
