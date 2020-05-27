
import pandas as pd
import pickle
import json
import numpy as np


def get_prob(path,test_x):

    with open(path, 'rb') as fr:
        model = pickle.load(fr)
    prob = model.predict_proba(test_x)
    return prob


def fun(x):
    if x == 0:
        return -1
    elif x == 1:
        return 1
    else:
        raise NameError



def preprocessing(json_path,df):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    label = data['label']
    feature = data['feature']   # 所有可能用到的特征

    df = df[feature]

    for i in feature:
        if i in label:
            continue
        else:
            data_ = data[i]   #[x_max,x_min,x_mean,x_std]
            # df[i] = (df[i] - data_[1]) / (data_[0]-data_[1])
            df[i] = (df[i] - data_[0]) / data_[1]

    X = np.array(df)
    return X


if __name__ == '__main__':
    filepath = 'Your test csv file path.'
    df=pd.read_csv(filepath,
               index_col=0,error_bad_lines=False, engine='python',encoding='gbk')

    all_model = {'rbf':'rbfsvm.pickle','linear':'linearsvm.pickle','poly':'polysvm.pickle','sigmoid':'sigmoidsvm.pickle'}
    filename = 'oldtmp2.json'
    modelpath = all_model['rbf']
    X = preprocessing(filename,df)
    n_samples, n_features = X.shape
    random_state = np.random.RandomState(0)
    X = np.c_[X, random_state.randn(n_samples, 2 * n_features)]
    p = X[0:10,:]
    prob = get_prob(modelpath,p)
    print(prob[:,1])




