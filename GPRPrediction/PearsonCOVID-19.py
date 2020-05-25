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
from scipy.stats import pearsonr
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=15)
warnings.filterwarnings("ignore")


item = ['Glucocorticoid', 'Anti-virus', 'Chloroquine', 'Antibiotic', 'Immunoglobulin', 'Traditional Chinese Medicine']
for itemindex in range(0, 6):
    med_kind = item[itemindex]
    ##数据初始化
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    df = pd.read_excel('D:\COVID-19\统计结果\相关性分析.xlsx', sheet_name= med_kind)


    x = np.asarray(df.the_area_ratio)
    # y = np.asarray(df.Medication_before)
    y = np.asarray(df.Medication_before)

    sns.regplot(x=x, y=y, color="red")
    # p4 = plt.scatter( x, y, s = 112, c='red',marker = ',', alpha=0.5,label='Senior')
    ps = pearsonr(x, y)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.yticks( size = 20)
    # plt.xticks( size = 20)
    plt.xlabel('Area_Ratio')
    plt.ylabel('Day_Ratio')
    plt.title(med_kind, fontproperties=font)
    plt.text(0.6*max(x), 0.8*max(y), r'$pcc='+str(round(ps[0],4))+'(p='+str(round(ps[1],4))+')$')
    # plt.legend( loc = 'lower right',fontsize=20)
    plt.savefig('D:\COVID-19\统计结果\\'+ med_kind +'.svg')
    plt.show()
    plt.close()


