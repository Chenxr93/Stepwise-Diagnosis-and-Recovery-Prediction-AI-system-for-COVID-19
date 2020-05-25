import numpy as np
import pandas as pd
import datetime
import math
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar, exp
from scipy.stats import pearsonr
import warnings
from matplotlib.font_manager import FontProperties
from sklearn.gaussian_process.kernels import RBF, DotProduct, ExpSineSquared, RationalQuadratic, Matern, \
    ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.spatial.distance import pdist


font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=15)
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


kernel = 1.0 * RBF(1.0)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
x_set = np.arange(1, 35, 1)
x_set = np.array([[i] for i in x_set])

# m = '谷草谷丙'
item = ['C-反应蛋白', '直接胆红素', '谷草转氨酶', '乳酸脱氢酶','淋巴细胞绝对值', '钾', '钙']
for itemindex in range(0, 7):
    m = item[itemindex]
    df = pd.read_excel('D:\COVID-19\检验报告\检验报告_' + m + '.xlsx')
    outdf = df[~df['num'].isin([1])]
    outx = outdf.姓名.unique()
    outy = range(1, 35)
    outex = pd.DataFrame(index=outx, columns=outy)
    outex.index.name = 'name'
    outex.columns.name = 'date'



    # unit = df.单位.loc[df['检验项目'] == m]

    for k,v in outdf.groupby(by='姓名'):
        xobs = np.array(outdf.new_date.loc[df.姓名 == k]).reshape(-1, 1)
        yobs = np.array(outdf.结果.loc[df.姓名 == k])
        # Fit the model to the data (optimize hyper parameters)
        gp.fit(xobs, yobs)
        # predictions
        means, sigmas = gp.predict(x_set, return_std=True)

        outex.loc[k, 0:len(means)] = means

        plt.figure(figsize=(8, 5))
        plt.errorbar(x_set, means, yerr=sigmas, alpha=0.5, ecolor='navajowhite', color='orange', label='error')
        plt.plot(x_set, means, 'orange', linewidth=4, label='predict')
        plt.plot(xobs, yobs, 'bo:', label='data')
        # plt.title(k, fontproperties=font)
        plt.xlabel("Date interval")
        plt.ylabel(m + '(' + unit[0] + ')', fontproperties=font)
        # plt.ylabel(m, fontproperties=font)
        plt.legend()
        plt.savefig("D:\COVID-19\检验报告\GPRfig\\" + m +'\\'+ str(k) + '_' + m + '.jpg', bbox_inches='tight')
        # plt.show()


    outex.to_excel('D:\COVID-19\检验报告\GPRfig\检验报告_'+ m +'序列.xlsx')