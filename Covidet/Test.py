import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, classification_report, roc_curve
import torch
import copy

from dealdata.data_loader_5_9 import DataBowl
from my_models import get_model
from myconfig import create_parser, convert_to_params


def eval(model, model_path, dl, phase='test', bs=96):
    model.cuda()
    #try:
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    #except:
    #    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("load model")
    pre = {"label":[], "pre_cls":[], "prob":[], "name":[]}

    for i, (xx, y, n) in enumerate(tqdm(dl)):
        logitsall = 0
        y = y.numpy().reshape(-1)
        tqdm.write("{}".format(y))
        m_batch = len(xx) // bs
        if len(xx) % bs != 0:
            m_batch += 1
        for j in range(m_batch):
            #pdb.set_trace()
            x = copy.deepcopy(xx[j*bs:(j+1)*bs])
            x = torch.cat(x,dim=0)
            x = x.cuda()
            with torch.no_grad():
                logits = model(x)
            #pdb.set_trace()
            # preds = preds.float().detach().cpu().numpy().reshape(-1)
            logitsall += logits.sum().detach().cpu().numpy().reshape(-1)
        logitsall = logitsall/len(xx)
        preds = logitsall > 0.5
        print(logitsall.shape,preds.shape,len(n))
        for one in range(preds.shape[0]):
            pre['label'].append(y[one])
            pre['pre_cls'].append(preds[one])
            pre['prob'].append(logitsall[one])
            pre['name'].append(n[one])
    tqdm.write("%d" % len(pre['label']))

    auc = roc_auc_score(pre['label'], pre['prob'])
    report = classification_report(pre['label'], pre['pre_cls'], target_names=['neg', 'pos'], digits=4)
    fpr, tpr, _ = roc_curve(pre['label'], pre['prob'])
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    #plt.show()
    #plt.savefig('./test_roc_5-8_clinical2.png')
    # np.savez_compressed('test_.npz', fpr=fpr, tpr=tpr,label=pre['label'], cls=pre['pre_cls'], prob=pre['prob'])
    print("AUC:\n{}".format(auc))
    print("report:\n{}".format(report))
    df = pd.DataFrame(pre)
    df.to_csv('test_result.csv', encoding='utf-8-sig', index=None)
    return pre


if __name__ == "__main__":
    flags = create_parser()
    hparams = convert_to_params(flags)

    os.environ['CUDA_VISIBLE_DEVICES'] =hparams['gpu_num']

    val_ds = DataBowl(phase='val',
                      path_val=hparams['Test_dir'],three=2)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, num_workers=16, shuffle=False)
    print(len(val_dl), len(val_ds))
    model = get_model('resGRU', nclass=1000, pretrained=False).cuda()
    model_path = hparams['model_path']
    print("model path is {}.".format(model_path))

    #pre = eval(model, model_path, dl)

    eval(model, model_path, val_dl, phase='val')
