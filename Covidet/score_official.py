import numpy as np
import pandas as pd
from sklearn import metrics


def ODIR_Metrics(gt_data, pr_data):
    th = 0.5
    gt = gt_data.flatten()
    pr = pr_data.flatten()
    tp =  np.sum( pr*gt > 0.5 )
    tn = np.sum(gt + pr*(1-gt)<0.5)
    fp = np.sum( (1-gt)*pr >0.5)
    fn = np.sum( gt*pr +(1-gt) <0.5)
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    prcls = pr>th
    fpr, tpr, thresholds = metrics.roc_curve(gt_data,prcls, pos_label=1)

    kappa = metrics.cohen_kappa_score(gt, pr > th)
    f1 = metrics.f1_score(gt, pr > th, average='micro')
    auc = metrics.roc_auc_score(gt, pr)
    final_score = (kappa + f1 + auc) / 3.0
    return kappa, f1, auc, final_score,sensitivity,specificity