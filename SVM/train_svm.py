import numpy as np
import pandas as pd
from sklearn import metrics
import pickle
from sklearn.externals import joblib
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import json
import codecs

def fun(x):
    if x == 0:
        return -1
    elif x == 1:
        return 1
    else:
        raise NameError

def getData(filepath,discrete_feature,continue_feature):

    df=pd.read_csv(filepath,
               index_col=0,error_bad_lines=False, engine='python',encoding='gbk')
    X_columns=[]
    for i in df.columns:
        if i in ['train','label','rbf','序号','名字','poly','sigmoid','linear']:
            continue
        else:
            X_columns.append(i)
            if i in discrete_feature:
                df[i].apply(lambda x: fun(x))

    test_data = df[df['train'] == 0]
    train_data = df[df['train'] == 1]

    train_x =train_data[X_columns]
    train_y = train_data[['label']]
    test_x =test_data[X_columns]
    test_y = test_data[['label']]

    mydict={'feature':X_columns,'label':discrete_feature}  # 名字 数组[] 不要的数据怎么处理？X_columns储存

    for i in X_columns:
        if i in discrete_feature:
            continue
        elif i in continue_feature:

            x_max = train_x[i].max()
            x_min = train_x[i].min()
            train_x[i] = (train_x[i] - x_min) / (x_max-x_min)
            # print('{}均值：{}'.format(i,x_mean))
            test_x[i] = (test_x[i] - x_min) / (x_max - x_min)
            # continue
            x_mean = train_x[i].mean()
            x_std = train_x[i].std()
            train_x[i] = (train_x[i] - x_mean) / x_std
            test_x[i] = (test_x[i] - x_mean) / x_std
            mydict[i] = [x_max,x_min,x_mean, x_std]
            # print('test {}均值：{}'.format(i,x_mean2))
        else:
            print('error1')
        if np.isnan(train_x[i]).any():
            print('error2')
        if np.isnan(test_x[i]).any():
            print('error3')
    with codecs.open("oldtmp2.json", 'w', encoding='utf-8') as f:
        json.dump(mydict, f, ensure_ascii=False)

    return np.array(train_x),np.array(train_y),np.array(test_x),np.array(test_y)


def train(modelnamem,train_x,train_y,test_x,random_state=np.random.RandomState(0),noise_alpha=1.55):
    classifier = svm.SVC(kernel=modelnamem, probability=True,random_state=random_state)

    n_samples, n_features = train_x.shape
    n_samples2, n_features2 = test_x.shape
    X = np.c_[train_x, random_state.randn(n_samples, int(noise_alpha * n_features))]
    X_valid = np.c_[test_x, random_state.randn(n_samples2, int(noise_alpha * n_features))]
    classifier = classifier.fit(X, train_y)

    train_probas = classifier.predict_proba(X)
    test_probas = classifier.predict_proba(X_valid)

    return classifier,train_probas,test_probas





def save_model(model,train_probas,test_probas,filepath,modelnamem):

    with open('{}_svm.pickle'.format(modelnamem), 'wb') as fw:
        pickle.dump(model, fw)
    df=pd.read_csv(filepath,
           index_col=0,error_bad_lines=False, engine='python',encoding='gbk')

    test_data = df[df['train'] == 0]
    train_data = df[df['train'] == 1]

    test_data[modelnamem] = test_probas[:, 1]
    train_data[modelnamem] = train_probas[:, 1]
    new_train_data = pd.concat([train_data, test_data])
    new_train_data.to_csv(filepath
                          , float_format='%.3f')
    return train_probas,test_probas




if __name__ == '__main__':
    filepath = 'Your csv file path'
    modelname = ['rbf','sigmoid','linear','poly']
    #modelname = ['linear' ]
    discrete_feature = [ 'Fever',	'Conjunctival congestion','Nasal congestion ' ,'Headache','Dry cough',
          'Pharyngalgia','Productive cough','Fatigue','Hemoptysis','Shortness of breath',	'Nausea/vomiting ' ,'Diarrhea ',
          'Myalgia/arthralgia ','Chill','Throat congestion','Tonsil swelling ','Enlargement of lymph nodes ','Rash',
          'Unconsciousness',	'Comorbidities','Any Comorbidities','COPD','Diabetes','Hypertension','Cardiovascular disease',
          'Cerebrovascular disease','Hepatitis B infection','Malignancy','Chronic kidney disease'] # 分类

    continue_feature=['Highest temperature (℃)','PaO2 (with oxygen inhalation), mmHg','leukocyte(*10^9/L)','Lymphocyte count（*10^9/L）','Platelet count(*10^9/L)',
        'Hemoglobin(g/dl)','Procalcitonin(ng/ml)','Lactate dehydrogenase(U/l)','Aspartate aminotransferase(U/l)',
        'Alanine aminotransferase(U/l)','Direct bilirubin(μmol/L)','Total bilirubin(μmol/l)','Creatine kinase(U/l)',
        'Creatinine(μmol/L/l)','Hypersensitive troponin I(pg/ml)','Albumin(g/L)','Sodium(mmol/l)','Potassium(mmol/l)','Chlorine(mmol/l)'] # 回归

    train_x,train_y,test_x,test_y = getData(filepath,discrete_feature,continue_feature)

    model,train_probas,test_probas = train(modelname[0],train_x,train_y,test_x)
    save_model(model,train_probas,test_probas,filepath,modelname[0])
    print(test_probas[0])


