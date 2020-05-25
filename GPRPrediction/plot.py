import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib.font_manager import FontProperties



font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=20)
font2 = {
         'weight': 'normal',
         'size': 20,
         }
df = pd.read_excel('D:\COVID-19\统计结果\柱状图.xlsx')


# l1 = np.array(df.iloc[1,1:])
# l2 = np.array(df.iloc[2,1:])
l1 = np.array(df.iloc[2,1:])

name=['Glucocorticoid', 'Anti-virus', 'Chloroquine', 'Antibiotic', 'Immunoglobulin', '   Traditional \nChinese \nMedicine']

total_width, n = 0.8, 3
width = total_width / n
x=[0,1,2,3,4,5]
plt.figure(figsize=(27, 9))
plt.rc('font', family='Arial', size=20, weight='normal')#设置中文显示，否则出现乱码！
# a=plt.bar(x, l1, width=width, label='3 CT scans', alpha=0.5)
# for i in range(len(x)):
#     x[i] = x[i] + width
# b=plt.bar(x, l2, width=width, label='4 CT scans',tick_label = name, alpha=0.5)
a=plt.bar(x, l1, width=width, tick_label = name, alpha=0.5)

# plt.xlabel("Class of medicines")
plt.ylabel("Effectiveness(%)")
#plt.title('')
# plt.legend()
plt.savefig('D:\COVID-19\统计结果\plot.svg')
plt.show()


