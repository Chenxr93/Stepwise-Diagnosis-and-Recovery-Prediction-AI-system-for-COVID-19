import os
import time
import math
import copy

import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn import DataParallel
import torch.nn.functional as F
from torch.autograd import Variable
import types
from dealdata.data_loader_2_24 import DataBowl
from lossfn import FocalLoss, focal_loss1
from my_models import get_model
from myconfig import create_parser, convert_to_params
from score_official import ODIR_Metrics

class savemodel(nn.Module):
    def __init__(self):
        super(savemodel, self).__init__()
        self.modelshare = None
        self.modellist = nn.ModuleList()
    def forward(self,x):
        return self.modellist[0](self.modelshare(x))
class Trainer(nn.Module):

    def __init__(self, hparams):
        super(Trainer, self).__init__()
        self.process = hparams['transform']
        self.hparams = hparams
        self.writer = SummaryWriter(comment=hparams['summary_comment'])
        self.fold_k = hparams['fold']
        self.lr = hparams['init_lr']
        self.best_models = {}
        np.random.seed(0)
        self.savapath = './savemodel_Cov3_22/'
        self.data_set = DataBowl(phase='train',path_train ='/data/chenxiangru/covidData/TrainCov3-22',
                path_val = '/data/chenxiangru/covidData/TestCov3-21')
        self.data_loader = torch.utils.data.DataLoader(self.data_set, shuffle=True,
                batch_size=self.hparams['tr_bs'],
                num_workers=8,
                )

        data_set_val = DataBowl(phase='val',path_train ='/data/chenxiangru/covidData/TrainCov3-22',
                path_val = '/data/chenxiangru/covidData/TestCov3-21')
        self.data_loader_val = torch.utils.data.DataLoader(data_set_val, shuffle=True,
                batch_size=self.hparams['tr_bs'],
                num_workers=8,
                                                           )
        self.tr_batch_nums = len(self.data_loader)
        self.total_tr_batch_nums = self.tr_batch_nums * hparams['epochs']
        self.model = get_model('resGRU', nclass=1000, pretrained=True)
        self.model = self.model.cuda()
        self.model = DataParallel(self.model)
        pre_dict = torch.load('/data/chenxiangru/covid2019/savemodel_Cov3_22/0.9570808967685236_0.8691860465116278_0.8691860465116279_0.8691860465116279_nvc.pkl')
        self.model.load_state_dict(pre_dict)
        self.criterion = nn.BCELoss()
        #self.criterion = nn.functional.cross_entropy
        #self.criterion = focal_loss1()

        #param_to_update = self.model.parameters()
        params = []
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-6)

    def save_model(self, checkdir, step=None):
        saveThemodel = savemodel()
        saveThemodel.modelshare = self.model['share']
        for i in self.model.keys():
            if i!='share':
                saveThemodel.modellist.append(self.model[i])
        save_name = os.path.join(checkdir, 'model.ckpt') + '-' + str(step)
        torch.save(saveThemodel.state_dict(), save_name)
        tqdm.write("Save model to {}.".format(save_name))
        return save_name

    def extract_model_spec(self, checkdir):
        self.model.load_state_dict(torch.load(checkdir))
        tqdm.write("Load model from {}.".format(checkdir))

    def eval_child_model(self, epoch):
        labelT = []
        pre_cls = []
        pre_score = []
        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(self.data_loader_val)):
                x = x.cuda()
                y = y.float().cuda()
                res = self.model(x)
                labelT.append(y.detach().cpu())
                pre_cls.append(res.view(-1).detach().cpu())
                if i % 10 == 0:
                    # tqdm.write("\033[1;33m Val acc is {} at {}\033[0m".format(accuracy_score(labelT, pre_cls), i))
                    tqdm.write("Label is {}".format(y))
                    tqdm.write("Pre is {}".format(res))
                labelT.append((1-y).detach().cpu())
                pre_cls.append((1.-res).view(-1).detach().cpu())

        labelT = torch.cat(labelT,dim = 0).numpy()
        pre_cls = torch.cat(pre_cls,dim = 0).numpy()

        #self.writer.add_scalar('training_loss', loss.data[0], n_iter)
        #self.writer.add_scalar('training_loss_{}'.format(t), loss_data[t], n_iter)
        kappa, f1, auc, avg ,sensi,spec= ODIR_Metrics(np.array(labelT), np.array(pre_cls))
        print(f1,auc,avg,sensi,spec)
        tqdm.write("F1 score is {}.\n".format(f1))
        tqdm.write("AUC is {}.\n".format(auc))
        tqdm.write("Kappa is {}.\n".format(kappa))
        tqdm.write("Avg is {}.\n".format(avg))
        tqdm.write("sensi is {}.\n".format(sensi))
        tqdm.write("spec is {}.\n".format(spec))
        self.to_sunmmary_writer('val', f1, auc, kappa, avg,sensi,spec, epoch)
        if not os.path.exists(self.savapath):
            os.mkdir(self.savapath)
        else:
            listDir = os.listdir(self.savapath)
            #for i in range(len(listDir)):
                #os.remove(os.path.join('./savemodelNP3_20',listDir[i]))

        ckpt = self.model.state_dict()
        torch.save(ckpt,self.savapath+'{}_{}_{}_{}_nvc.pkl'.format(auc, f1,sensi,spec))

        # if avg >= 0.66:
        #     self.best_models['{}_{}_{}.pkl'.format(self.fold_k, avg, f1)] = copy.deepcopy(self.model.state_dict())
        #     self.keep_models()
        return f1

    def _compute_final_accuracies(self, iteration):
        """Run once training is finished to compute final test accuracy."""
        if (iteration >= self.hparams['epochs'] - 1):
            test_accuracy = self.eval_child_model(self.data_loader_val)
        else:
            test_accuracy = 0
        return test_accuracy

    def _run_training_loop(self, curr_epoch):
        """Trains the model `m` for one epoch."""
        start_time = time.time()
        labelT = []
        pre_cls = []
        hard_example_x = []
        hard_example_y = []
        self.model.train()
        for i, (x, y) in enumerate(tqdm(self.data_loader)):
            if i>170:
                break
            x = x.cuda()
            y = y.float().cuda()
            #print(x.size(),y.size())
            lr_to_log = self.adjust_lr(i, curr_epoch)
            self.writer.add_scalar("train/lr", lr_to_log, curr_epoch)
            criterion_0 = self.criterion
            self.optimizer.zero_grad()
            x = Variable(x)
            res = self.model(x)
            labelT.append(y.detach().cpu())
            pre_cls.append(res.detach().cpu())
            # Scaled back-propagation
            self.optimizer.zero_grad()
            if i%100 == 0:
                print(res,y)
            res = res.view(-1)
            y = y.view(-1)
            loss =  criterion_0(res, y)
            loss.backward()
            self.optimizer.step()
            '''
            hard example mining
            '''
            # idcs = calculateHardFactor(preds, y)
            # hard_example_x.append(torch.index_select(x, 0, idcs).cpu())
            # hard_example_y.append(torch.index_select(y,0,idcs).cpu())
            # if len(hard_example_x) == x.size(0):
            #     del(x)
            #     del(y)
            #     hard_example_x = torch.cat(hard_example_x,dim = 0).cuda()
            #     hard_example_y = torch.cat(hard_example_y,dim = 0).cuda()
            #     logits = self.model(hard_example_x)
            #     preds = torch.sigmoid(logits.view(logits.size(0), -1))
            #     criterion_0 = self.criterion
            #     loss = criterion_0(preds, hard_example_y)
            #     loss = loss.mean()
            #     # print("loss is: ", loss)
            #     loss.backward()
            #     self.optimizer.step()
            #     del(hard_example_x)
            #     del(hard_example_y)
            #     hard_example_x = []
            #     hard_example_y = []
        pre_cls = torch.cat(pre_cls,dim = 0)
        labelT = torch.cat(labelT,dim = 0)
        print(pre_cls.size())
        print(labelT.size())
        pre_cls = pre_cls.numpy()
        labelT= labelT.numpy()
        kappa, f1, auc, avg,sensi,spec = ODIR_Metrics(np.array(labelT), np.array(pre_cls))
        self.writer.add_scalar("train/loss", loss, curr_epoch)
        tqdm.write("F1 score is {}.\n".format(f1))
        tqdm.write("AUC is {}.\n".format(auc))
        tqdm.write("Kappa is {}.\n".format(kappa))
        tqdm.write("Avg is {}.\n".format(avg))
        tqdm.write("sensi is {}.\n".format(sensi))
        tqdm.write("spec is {}.\n".format(spec))
        self.to_sunmmary_writer('train', f1, auc, kappa, avg,sensi,spec, curr_epoch)
        tqdm.write('Epoch time(min): {} at epoch {}.\n'.format(
            (time.time() - start_time) / 60.0, curr_epoch))
        return f1

    def run_model(self, epoch):
        """Trains and evalutes the image model."""

        valid_accuracy = self.eval_child_model(epoch)

        tqdm.write("\033[1;35m One epoch valed and acc is{}.\033[0m".format(valid_accuracy))
        return training_accuracy, valid_accuracy

    def reset_config(self, new_hparams):
        self.hparams = new_hparams
        self.data_set.reset_policy(new_hparams['hp_policy'])
        self.data_loader = torch.utils.data.DataLoader(self.data_set, shuffle=True,
                                                       batch_size=self.hparams['tr_bs'],
                                                       num_workers=0)

    def adjust_lr(self, step, epoch, gamma=2.0):
        if epoch < 5:
            lr = 0.001
        elif epoch <10:
            lr = 0.0001
        elif epoch < 15:
            lr = 0.0001
        else:
            lr = 0.00005
        #if epoch < 3:
        #    lr = self.lr * (epoch + 1) / 3
        #else:
            #current_step = epoch * self.tr_batch_nums + step
            #current_step = current_step - self.tr_batch_nums * 3
            #lr = self.lr * (1 + math.cos(math.pi * current_step / self.total_tr_batch_nums))/gamma
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr
        return lr

    def to_sunmmary_writer(self, phase, f1, auc, kappa, avg,sensi=0,spec = 0, epoch=0):
        self.writer.add_scalar("{}/f1".format(phase), f1, epoch)
        self.writer.add_scalar("{}/auc".format(phase), auc, epoch)
        self.writer.add_scalar("{}/kappa".format(phase), kappa, epoch)
        self.writer.add_scalar("{}/avg".format(phase), avg, epoch)
        self.writer.add_scalar("{}/avg".format(phase), sensi, epoch)
        self.writer.add_scalar("{}/avg".format(phase), spec, epoch)

    def keep_models(self):
        if len(self.best_models) <= 2:
            return
        else:
            key_list = sorted(list(self.best_models.keys()))
            for k in key_list[:-2]:
                del self.best_models[k]
        return

    def save_best_models(self):
        for k in self.best_models:
            dir_name = os.path.join('./models', str(self.fold_k))
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            fp = os.path.join(dir_name, k)
            torch.save(self.best_models[k], fp)
            tqdm.write("Save model to {}".format(fp))

    def close_writer(self):
        self.writer.export_scalars_to_json('./log.json')
        self.writer.close()

def calculateHardFactor(outputs, targets):
    factor = torch.abs(outputs - targets)
    factor = torch.sum(factor,dim=0)
    _,idcs = torch.topk(factor,1)
    return idcs




def main(fold=0):
    flags = create_parser()
    hparams_config = convert_to_params(flags)
    hparams_config['fold'] = fold
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_num

    trainer = Trainer(hparams_config)

    for i in trange(hparams_config['epochs']):
        train_acc, val_acc = trainer.run_model(i)
    trainer.save_best_models()
    trainer.close_writer()
    del(trainer)
    return

if __name__ == "__main__":
    for i in range(1):
        main(i)
