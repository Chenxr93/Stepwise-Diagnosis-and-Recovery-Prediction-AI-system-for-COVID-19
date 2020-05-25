import numpy as np
import pandas as pd
import warnings
from matplotlib.font_manager import FontProperties



font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=15)
warnings.filterwarnings("ignore")


of = pd.read_excel('D:\GPR\检验报告Pearson统计结果.xlsx')
item = ['白细胞', '血小板压积', '红细胞数', '血小板数', '中性粒细胞绝对值', '淋巴细胞绝对值', '血红蛋白浓度', '氯', '钾', '钠', '钙', '尿素氮', '肌酐', '谷草转氨酶', '谷丙转氨酶', '直接胆红素', '间接胆红素', '总胆红素', '总蛋白', '白蛋白', '球蛋白', '白球比例', '乳酸脱氢酶', '总胆汁酸', 'γ-谷氨酰转移酶', '碱性磷酸酶', '尿酸', '降钙素原(PCT)', 'C-反应蛋白', 'D-二聚体含量', '肌酸激酶', '肌红蛋白', '谷草谷丙']
for itemindex in range(0, 33):
    m = item[itemindex]
    df = pd.read_excel('D:\GPR\检验报告_' + m + '.xlsx')
    a =  list(df.groupby(by='姓名').pearson.unique())
    of.总数.loc[of['项目']==m] = len(a)
    of.强正相关.loc[of['项目']==m] = np.sum(list(map(lambda x: x > 0.9, a)))
    of.强负相关.loc[of['项目']==m] = np.sum(list(map(lambda x: x < -0.9, a)))
    of.正相关比例.loc[of['项目']==m] = np.sum(list(map(lambda x: x > 0, a)))/len(a)
    of.负相关比例.loc[of['项目']==m] = np.sum(list(map(lambda x: x < 0, a)))/len(a)
    of.强正相关比例.loc[of['项目'] == m] = np.sum(list(map(lambda x: x > 0.9, a))) / len(a)
    of.强负相关比例.loc[of['项目'] == m] = np.sum(list(map(lambda x: x < -0.9, a))) / len(a)








of.to_excel('D:\GPR\检验报告Pearson统计结果.xlsx')