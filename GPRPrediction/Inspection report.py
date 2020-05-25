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

df = pd.read_excel('D:\GPR\检验报告处理结果2020.5.1.xlsx')
of = pd.read_excel('D:\GPR\CT结果_2020.4.29.xlsx')

df.sort_values(by = ["患者ID", "检验项目", "测定时间"], inplace=True)

item = ['白细胞', '血小板压积', '红细胞数', '血小板数', '中性粒细胞绝对值', '淋巴细胞绝对值', '血红蛋白浓度', '氯', '钾', '钠', '钙', '尿素氮', '肌酐', '谷草转氨酶', '谷丙转氨酶', '直接胆红素', '间接胆红素', '总胆红素', '总蛋白', '白蛋白', '球蛋白', '白球比例', '乳酸脱氢酶', '总胆汁酸', 'γ-谷氨酰转移酶', '碱性磷酸酶', '尿酸', '降钙素原(PCT)', 'C-反应蛋白', 'D-二聚体含量', '肌酸激酶', '肌红蛋白']
for itemindex in range(0, 31):
    m = item[itemindex]
    unit = df.单位.loc[df['检验项目'] == m]

    for k,v in df.groupby(by='姓名'):
        result = v.结果.loc[v['检验项目'] == m]
        df.num.loc[(df['姓名']==k) & (df['检验项目']==m)] = result.shape[0]
        date = v.new_date.loc[v['检验项目'] == m]

        CTdate = of.new_date.loc[of['name']==k]
        CTdata = of.pnum_ratio.loc[of['name'] == k]



        if len(result) == 0:
            continue
        elif len(result) == 1:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            lns1 = ax.plot(date, result, 'ro:', label=m)

            ax2 = ax.twinx()
            lns2 = ax2.plot(CTdate, CTdata, 'bs:', label='CT')

            lns = lns1 + lns2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, prop=font, bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
            ax.set_xlabel("Date interval")
            ax.set_ylabel(m + '(' + unit.unique()[0] + ')', fontproperties=font)
            # ax.set_ylabel(m, fontproperties=font)
            ax2.set_ylabel('Area Ratio')
            # plt.savefig("D:\GPR\datafig\ " + str(k) + '_' + '谷草谷丙' + '.jpg', bbox_inches='tight')
            plt.savefig("D:\GPR\datafig\\" + m +'\\'+ str(k) + '_' + m + '.jpg', bbox_inches='tight')
            # plt.show()
        else:
            new_x = list(set(np.array(date).astype(int)).union(set(np.array(CTdate).astype(int))))
            new_x = np.sort(new_x)
            new_y1 = np.interp(new_x, np.array(date).astype(int), np.array(result).astype(float))
            new_y2 = np.interp(new_x, np.array(CTdate).astype(int), np.array(CTdata).astype(float))


            ps = pearsonr(new_y1, new_y2)
            df.pearson.loc[(df['姓名'] == k) & (df['检验项目'] == m)] = ps[0]
            # plt.figure(figsize=(8, 5))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            lns1 = ax.plot(date, result, 'ro:', label=m)
            ax.plot(new_x, new_y1, 'rx:', label=m+'插值')
            ax2 = ax.twinx()
            lns2 = ax2.plot(CTdate, CTdata, 'bs:', label='CT')
            ax2.plot(new_x, new_y2, 'bx:', label='CT插值')
            lns = lns1 + lns2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs,prop=font, bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
            ax.set_xlabel("Date interval")
            ax.set_ylabel(m + '(' + unit.unique()[0] + ')', fontproperties=font)
            # ax.set_ylabel(m , fontproperties=font)
            ax2.set_ylabel('Area Ratio')
            plt.text(min(new_x), min(new_y2), r'$pcc =' + str(round(ps[0], 4)) + '$')
            # plt.savefig("D:\GPR\datafig\ " + str(k) + '_' + '谷草谷丙' + '.jpg', bbox_inches='tight')
            plt.savefig("D:\GPR\datafig\\" + m +'\\' + str(k) + '_' + m + '.jpg', bbox_inches = 'tight')
            # plt.show()





    # df.sort_values(by = ["患者ID", "num", "检验项目", "测定时间"], inplace=True)
    # df3=df.drop('num'==1,axis=0)


    # a=df.groupby('患者ID').检验项目.value_counts()

    df.to_excel('D:\GPR\检验报告_'+ m +'.xlsx')