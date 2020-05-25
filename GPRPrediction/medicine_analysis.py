import numpy as np
import pandas as pd
import datetime
import math
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar, exp
import warnings
from matplotlib.font_manager import FontProperties
from sklearn.gaussian_process.kernels import RBF,DotProduct,ExpSineSquared,RationalQuadratic,Matern, ConstantKernel as C
from sklearn.gaussian_process import GaussianProcessRegressor
from collections import  Counter

font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=15)
warnings.filterwarnings("ignore")

## 高斯模型参数设置
# kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (0.5, 2))
kernel = 1.0 * RBF(1.0)
# kernel = RationalQuadratic(alpha=1, length_scale=1.5)
# kernel = Matern(length_scale=1, nu=1.5)
# kernel = ExpSineSquared(length_scale=1, periodicity=1.5)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
#高斯过程X轴点分布
# x_set = np.arange(-29, 61, 0.1)
# x_set = np.array([[i] for i in x_set])
x_set1 = np.arange(-29, 61, 1)
x_set1 = np.array([[i] for i in x_set1])


## 数据初始化
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df = pd.read_excel('D:\COVID-19\CT结果_2020.4.29.xlsx')
of = pd.read_excel('D:\COVID-19\激素_2020.4.29.xlsx')

#CT新增数据
df.loc[:, 'the_most_serious_day'] = 1
df.loc[:, 'the_area_ratio'] = 0
df.loc[:, 'Medication_before'] = 0
df.loc[:, 'Medication_after'] = 0
df.loc[:, 'Medication_total'] = 0

#医嘱新增数据
of.loc[:, 'the_most_serious_day'] = 1
of.loc[:, 'the_area_ratio'] = 0
of.loc[:, 'Medication_before'] = 0
of.loc[:, 'Medication_after'] = 0
of.loc[:, 'Medication_total'] = 0
of.loc[:, 'gradient_before'] = 0
of.loc[:, 'gradient_after'] = 0
of.loc[:, 'gradient_diff'] = 0
of.loc[:, 'gradient_trend'] = 0

#参数设置
medicine = ['激素', '抗生素', '抗病毒', '中药', '抗疟药物', '免疫球蛋白', '干扰素']
color = ['bs:', 'gv:', 'y^:', 'cx:', 'kD:', 'mp:','ro:']
medindex = 0

#归一化数据，计算正态分布参数
for k,v in df.groupby(by='name'):

    # Some data
    xobs = np.array(df.new_date.loc[df.name == k]).reshape(-1,1)
    yobs = np.array(df.pnum_ratio.loc[df.name == k])

    # Fit the model to the data (optimize hyper parameters)
    gp.fit(xobs, yobs)
    # predictions
    # means, sigmas = gp.predict(x_set, return_std=True)
    means1, sigmas1 = gp.predict(x_set1, return_std=True)


    #患病面积最大及天数
    maxarea = max(means1)
    bb = means1.tolist()
    maxday = bb.index(max(bb)) - 29

    df.the_area_ratio.loc[df.name == k] = maxarea
    df.the_most_serious_day.loc[df.name == k] = maxday
    of.the_area_ratio.loc[of.姓名 == k] = maxarea
    of.the_most_serious_day.loc[of.姓名 == k] = maxday


    x1obs = np.array(of.new_date.loc[of.姓名 == k]).reshape(-1, 1)

    ofindex = np.array(of.new_date.loc[of.姓名 == k]) + 29         #医嘱时间在高斯过程中的索引
    gradtestf = means1[ofindex] - means1[ofindex - 1]              #前一天梯度
    gradtestl = means1[ofindex + 1] - means1[ofindex]              #后一天梯度

    of.gradient_before.loc[of.姓名 == k] = gradtestf
    of.gradient_after.loc[of.姓名 == k] = gradtestl
    of.gradient_diff.loc[of.姓名 == k] = gradtestl - gradtestf     #梯度变化
    g_t = gradtestl - gradtestf > 0                                #梯度趋势：升1降平0
    of.gradient_trend.loc[of.姓名 == k] = g_t

    y1obs = maxarea / 4 * g_t


    M_b = int(sum(np.where(x1obs <= maxday,1,0)))
    M_a = int(sum(np.where(x1obs > maxday,1,0)))
    M_t = len(x1obs)


    df.Medication_before.loc[df.name == k] = M_b
    df.Medication_after.loc[df.name == k] = M_a
    df.Medication_total.loc[df.name == k] = M_t
    of.Medication_before.loc[of.姓名 == k] = M_b
    of.Medication_after.loc[of.姓名 == k] = M_a
    of.Medication_total.loc[of.姓名 == k] = M_t


    plt.figure(figsize=(8, 5))
    plt.errorbar(x_set1, means1, yerr=sigmas1, alpha=0.5, ecolor='navajowhite', color='orange', label='error')
    plt.plot(x_set1, means1, 'orange', linewidth=4, label='predict')
    plt.plot(xobs, yobs, 'ro:', label='data')
    plt.plot(maxday, maxarea, 'yo:', label='max')
    plt.plot(x1obs, y1obs, color[medindex], label=medicine[medindex])
    # plt.title(k, fontproperties=font)
    plt.xlabel("Date interval")
    plt.ylabel("Area ratio")
    plt.legend(prop=font,bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.savefig("D:\COVID-19\datafig\ " + str(k) + '_' + medicine[medindex] + '.jpg')
    plt.show()



df.to_excel('D:\COVID-19\CT数据统计_2020.4.29.xlsx')
of.to_excel('D:\COVID-19\医嘱斜率统计_2020.4.29.xlsx')
