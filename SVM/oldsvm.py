import numpy as np
import pandas as pd
from sklearn import metrics

import pickle
from sklearn.externals import joblib

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


df=pd.read_csv('E:\\Desktop\\kaggle\\RandomForestClassifier\\sub_svm_out_withnoise2.csv',
               index_col=0,error_bad_lines=False, engine='python',encoding='gbk')



def fun(x):
    if x == 0:
        return -1
    elif x == 1:
        return 1
    else:
        raise NameError

label = [ 'Fever',	'Conjunctival congestion','Nasal congestion ' ,'Headache','Dry cough',
          'Pharyngalgia','Productive cough','Fatigue','Hemoptysis','Shortness of breath',	'Nausea/vomiting ' ,'Diarrhea ',
          'Myalgia/arthralgia ','Chill','Throat congestion','Tonsil swelling ','Enlargement of lymph nodes ','Rash',
          'Unconsciousness',	'Comorbidities','Any Comorbidities','COPD','Diabetes','Hypertension','Cardiovascular disease',
          'Cerebrovascular disease','Hepatitis B infection','Malignancy','Chronic kidney disease'] # 分类


label2=['Highest temperature (℃)','PaO2 (with oxygen inhalation), mmHg','leukocyte(*10^9/L)','Lymphocyte count（*10^9/L）','Platelet count(*10^9/L)',
        'Hemoglobin(g/dl)','Procalcitonin(ng/ml)','Lactate dehydrogenase(U/l)','Aspartate aminotransferase(U/l)',
        'Alanine aminotransferase(U/l)','Direct bilirubin(μmol/L)','Total bilirubin(μmol/l)','Creatine kinase(U/l)',
        'Creatinine(μmol/L/l)','Hypersensitive troponin I(pg/ml)','Albumin(g/L)','Sodium(mmol/l)','Potassium(mmol/l)','Chlorine(mmol/l)'] # 回归

print(df.duplicated().any)
print(df.info())
X_columns=[]
for i in df.columns:
    if i in ['train','label','rbf','序号','名字','poly','sigmoid','linear']:
        continue
    elif i=='任何其他合并症':
        continue
    else:
        X_columns.append(i)
        if i in label:
            df[i].apply(lambda x: fun(x))

print(len(X_columns))

test_data = df[df['train'] == 0]
train_data = df[df['train'] == 1]


train_x =train_data[X_columns]
train_y = train_data[['label']]
test_x =test_data[X_columns]
test_y = test_data[['label']]

mydict={'feature':X_columns,'label':label}  # 名字 数组[] 不要的数据怎么处理？X_columns储存
for i in X_columns:
    if i in label:
        continue
    elif i in label2:
        # x_mean = train_x[i].mean()
        # x_std = train_x[i].std()
        # train_x[i] = (train_x[i] - x_mean) / x_std
        # x_mean2 = test_x[i].mean()
        # x_std2 = test_x[i].std()
        # test_x[i] = (test_x[i] - x_mean2) / x_std2

        x_max = train_x[i].max()
        x_min = train_x[i].min()
        train_x[i] = (train_x[i] - x_min) / (x_max-x_min)
        # print('{}均值：{}'.format(i,x_mean))
        x_max2 = test_x[i].max()
        x_min2 = test_x[i].min()
        test_x[i] = (test_x[i] - x_min2) / (x_max2 - x_min2)
        # continue
        x_mean = train_x[i].mean()
        x_std = train_x[i].std()
        train_x[i] = (train_x[i] - x_mean) / x_std
        x_mean2 = test_x[i].mean()
        x_std2 = test_x[i].std()
        test_x[i] = (test_x[i] - x_mean2) / x_std2
        mydict[i] = [x_max,x_min,x_mean, x_std]
        # print('test {}均值：{}'.format(i,x_mean2))
    else:
        print('error1')
    if np.isnan(train_x[i]).any():
        print('error2')
    if np.isnan(test_x[i]).any():
        print('error3')
    # set parameter
    # train_x[i] = (train_x[i] - x_min) / (x_max - x_min)

import json
# with open('data.json', 'w', encoding='utf-8') as fs:
#     json.dump(mydict, fs)
import codecs
with codecs.open("oldtmp2.json", 'w', encoding='utf-8') as f:
    json.dump(mydict, f, ensure_ascii=False)



X = np.array(train_x)
y = np.array(train_y)
X_valid = np.array(test_x)
Y_valid = np.array(test_y)

model__ = 'linear'
cv = StratifiedKFold(n_splits=6)
#### 噪声版本，不用归一化
random_state = np.random.RandomState(0)
classifier = svm.SVC(kernel=model__, probability=True,random_state=random_state)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

n_samples, n_features = X.shape
n_samples2, n_features2 = X_valid.shape
X = np.c_[X, random_state.randn(n_samples, int(1.55 * n_features))]
X_valid = np.c_[X_valid, random_state.randn(n_samples2, int(1.55 * n_features))]
i = 0




for train, test in cv.split(X, y):
    probas_ = classifier.fit(X, y).predict_proba(X_valid)
    print('************')
    probas_2 = classifier.predict_proba(X)

    # 模型保存
    # Compute ROC curve and area the curve
    # 通过roc_curve()函数，求出fpr和tpr，以及阈值
    fpr, tpr, thresholds = roc_curve(Y_valid, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    i += 1

    with open('{}svm2.pickle'.format(model__), 'wb') as fw:
        pickle.dump(classifier, fw)

    test_data['prb_covid'] = classifier.predict_proba(X_valid)[:, 1]
    test_data['train'] = 0

    train_data['prb_covid'] = classifier.predict_proba(X)[:, 1]
    train_data['train'] = 1
    new_train_data = pd.concat([train_data, test_data])

    new_train_data.to_csv('E:\\Desktop\\kaggle\\RandomForestClassifier\\{}final_svm_out2.csv'.format(model__)
                          , float_format='%.3f')
    break
    # df['train2'][train] = 1
    # df['train2'][test] = 0
    # df['prb_covid2'][train] = probas_2
    # if i==1:
    #     df['prb_covid2'] = probas_2[:,1]
    #     df['train2'] = df.apply(lambda  x :0 if x.prb_covid2 in probas_ else 1, axis=1)

# 画对角线
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)  # 在mean_fpr100个点，每个点处插值插值多次取平均
mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
# 画平均ROC曲线

std_auc = np.std(aucs, axis=0)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (area = %0.2f)' % mean_auc, lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

#plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# features = X_columns
# importances = classifier.coef_[0]
# importances = np.abs(importances)
# importances = importances/np.max(importances)
# indices = np.argsort(importances)
# num_features = len(importances)
#
# out_list = []
# out_feature = []
# 将特征重要度以柱状图展示
# others = 0
# for i in indices:
#     if importances[i]<0.06:
#         others+=importances[i]
#     else:
#         out_list.append(importances[i])
#         out_feature.append(features[i])
# out_list.append(others)
# out_feature.append('other')




colors =[
    'pink',
    'deeppink',
    'violet',
    'darkviolet',
    'mediumpurple',
    'blue',
    'royalblue',
    'skyblue',
    'deepskyblue',
    'cyan',
    'palegreen',
    'chartreuse',
    'yellow',
    'gold',
    'orange',
    'orangered',
    'r',
    'lightgray'
]
# #explode=(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.05)
# # plt.pie(importances[indices],labels=[features[i] for i in indices],autopct='%1.1f%%',shadow=False,startangle=150)
# plt.pie(out_list,labels=out_feature,colors = colors,shadow=False,startangle=150)
# plt.axis('equal')
# plt.show()

# C，Kernel，gama，
# params=classifier.get_params()
# aaa = classifier.support_vectors_
# w1=classifier.coef_
# w2=classifier.coef_[:,1]
# print(w1)
# 特征重要度
# features = X_columns
# importances = classifier.feature_importances_
# indices = np.argsort(importances)[::-1]
# num_features = len(importances)
#
# # 将特征重要度以柱状图展示
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(num_features), importances[indices], color="g", align="center")
# plt.xticks(range(num_features), [features[i] for i in indices], rotation='45')
# plt.xlim([-1, num_features])
# plt.show()
#