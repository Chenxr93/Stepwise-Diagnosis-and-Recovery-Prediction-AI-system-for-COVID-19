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

## data processing
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df = pd.read_excel('D:\GMM\GPR\GPtest.xlsx')
datafig_path_w = 'D:\GMM\GPR\wrongdatafig\ '
datafig_path_c = 'D:\GMM\GPR\wrongdatafig\ '




def dfdiv(a, b):
    return a / b
df.loc[:, 'pnum_ratio'] = df.apply(lambda row: dfdiv(row['disease_pixels'], row['lobe_pixels']), axis = 1)
df.loc[:, 'date'] = pd.to_datetime(df['date'], format = '%Y/%m/%d')
df.loc[:, 'new_date'] = datetime.timedelta(days=1)
df.loc[:, 'correction'] = 100
df.loc[:, 'picnum'] = 0
df.loc[:, 'range'] = 100
df.sort_values(by = ["name", "date"], inplace=True)
lack1df = df.copy()

#output settings
outdf = pd.read_excel('D:\GMM\GPR\\train\GPtrain.xlsx')
outx = outdf.name.unique()
outy = range(1,91)                     #predict days
outex = pd.DataFrame(index = outx,columns = outy)
outex.index.name = 'name'
outex.columns.name = 'date'

lack1outex = outex.copy()


# Plot points
x_set = np.arange(-29, 61, 0.1)
x_set = np.array([[i] for i in x_set])
x_set1 = np.arange(-29, 61, 1)
x_set1 = np.array([[i] for i in x_set1])
countnum = 0
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=15)

## GPR kernel setting
kernel = 1.0 * RBF(1.0)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)

#GPR main program
for k,v in df.groupby(by='name'):
    indj = 0
    indk = 90
    lack1indj = 0
    lack1indk = 90
    df.new_date.loc[df.name == k] = v.loc[:, 'date']-min(v.loc[:, 'date']) + datetime.timedelta(days=1)    #Using the first CT date as the first day
    df.picnum.loc[df.name == k] = v.shape[0]
    lack1df.picnum.loc[df.name == k] = v.shape[0]
    # GPR input data
    xobs = np.array(df.new_date.loc[df.name == k].dt.days).reshape(-1,1)
    yobs = np.array(df.pnum_ratio.loc[df.name == k])

    obsnum = 3    # Using CT data nums
    if len(xobs) < obsnum:
        obsnum = len(xobs)

    # Select the first few points
    lack1xobs = xobs[0:obsnum]
    lack1yobs = yobs[0:obsnum]

    newxobs = xobs
    newlack1xobs = lack1xobs

    # Fit the model to the data (optimize hyper parameters)
    gp.fit(xobs, yobs)
    # predictions
    means, sigmas = gp.predict(x_set, return_std=True)
    means1, sigmas1 = gp.predict(x_set1, return_std=True)

    areas = means1
    areasigmas = sigmas1
    areasx_set = np.arange(1, len(areas)+1, 1)
    areasx_set = np.array([[i] for i in areasx_set])
    outex.loc[k, 0:len(areas)] = areas
    print(u'name:', k)
    print(u'indj:', indj)
    print(u'indk:', indk)

    if max(areas)>=0.5:
        countnum = countnum + 1
    print(countnum)


    # Fit the model to the data (optimize hyper parameters)
    gp.fit(lack1xobs, lack1yobs)
    # predictions
    lack1means, lack1sigmas = gp.predict(x_set, return_std=True)
    lack1means1, lack1sigmas1 = gp.predict(x_set1, return_std=True)


    lack1areas = lack1means1
    lack1areasigmas = lack1sigmas1
    lack1areasx_set = np.arange(1, len(lack1areas) + 1, 1)
    lack1areasx_set = np.array([[i] for i in lack1areasx_set])
    lack1outex.loc[k, 0:len(lack1areas)] = lack1areas

    dateindex = np.array(df.new_date.loc[df.name == k].dt.days) - 1 + 30


    if obsnum>=len(dateindex):
        lack1df.correction.loc[df.name == k] = 101
        df.correction.loc[df.name == k] = 101

    else:
        theday = dateindex[-1]       #Using the last CT date as the predict day

        designrange = 15
        lackrange = designrange
        lowrange = theday - designrange
        if len(lack1areas) - theday < designrange:
            lackrange = len(lack1areas) - theday
        if theday - designrange < 0:
            lowrange = 0
        sym = 0

        thedayval = areas[theday]
        threshold = 0.008       #An acceptable data error
        for i in reversed(range(lowrange, theday + lackrange)):
            if thedayval > max(lack1areas[lowrange:theday + lackrange]) or thedayval < min(lack1areas[lowrange:theday + lackrange]):
                if (lack1areas[i] - thedayval + threshold) * (lack1areas[i - 1] - thedayval - threshold) <= 0 or\
                (lack1areas[i] - thedayval - threshold) * (lack1areas[i - 1] - thedayval + threshold) <= 0 :
                    dayrange = i - theday
                    sym = 1
                    print(u'dayrange:', dayrange)
                    lack1df.correction.loc[df.name == k] = dayrange
                    lack1df.range.loc[df.name == k] = abs(dayrange)
                    df.range.loc[df.name == k] = abs(dayrange)
                    df.correction.loc[df.name == k] = 1

                    plt.figure(figsize=(8, 5))
                    plt.errorbar(x_set, means, yerr=sigmas, alpha=0.5, ecolor='navajowhite', color='orange', label='error')
                    plt.plot(x_set, means, 'orange', linewidth=4, label='predict')
                    plt.plot(xobs, yobs, 'bo:', label='data')
                    plt.xlabel("Date interval")
                    plt.ylabel("Area ratio")
                    plt.legend()
                    plt.savefig(datafig_path_c + str(k) + '.jpg')
                    plt.show()

                    plt.figure(figsize=(8, 5))
                    plt.errorbar(x_set, lack1means, yerr=lack1sigmas, alpha=0.5, ecolor='navajowhite', color='orange',
                                 label='error')
                    plt.plot(x_set, lack1means, 'orange', linewidth=4, label='predict')
                    plt.plot(lack1xobs, lack1yobs, 'bo:', label='data')
                    plt.xlabel("Date interval")
                    plt.ylabel("Area ratio")
                    plt.legend()
                    plt.savefig(datafig_path_c + str(k) + 'lack1.jpg')
                    plt.show()
                    break
            elif (lack1areas[i] - thedayval) * (lack1areas[i - 1] - thedayval) <= 0:
                    dayrange = i - theday
                    sym = 1
                    print(u'dayrange:', dayrange)
                    lack1df.correction.loc[df.name == k] = dayrange
                    lack1df.range.loc[df.name == k] = abs(dayrange)
                    df.range.loc[df.name == k] = abs(dayrange)
                    df.correction.loc[df.name == k] = 1

                    plt.figure(figsize=(8, 5))
                    plt.errorbar(x_set, means, yerr=sigmas, alpha=0.5, ecolor='navajowhite', color='orange', label='error')
                    plt.plot(x_set, means, 'orange', linewidth=4, label='predict')
                    plt.plot(xobs, yobs, 'bo:', label='data')
                    plt.xlabel("Date interval")
                    plt.ylabel("Area ratio")
                    plt.legend()
                    plt.savefig(datafig_path_c + str(k) + '.jpg')
                    plt.show()

                    plt.figure(figsize=(8, 5))
                    plt.errorbar(x_set, lack1means, yerr=lack1sigmas, alpha=0.5, ecolor='navajowhite', color='orange',
                                 label='error')
                    plt.plot(x_set, lack1means, 'orange', linewidth=4, label='predict')
                    plt.plot(lack1xobs, lack1yobs, 'bo:', label='data')
                    plt.xlabel("Date interval")
                    plt.ylabel("Area ratio")
                    plt.legend()
                    plt.savefig(datafig_path_c + str(k) + 'lack1.jpg')
                    plt.show()
                    break


        if sym == 0:
            plt.figure(figsize=(8, 5))
            plt.errorbar(x_set, means, yerr=sigmas, alpha=0.5, ecolor='navajowhite', color='orange', label='error')
            plt.plot(x_set, means, 'orange', linewidth=4, label='predict')
            plt.plot(xobs, yobs, 'bo:', label='data')
            plt.xlabel("Date interval")
            plt.ylabel("Area ratio")
            plt.legend()
            plt.savefig(datafig_path_w + str(k) + '.jpg')
            plt.show()

            plt.figure(figsize=(8, 5))
            plt.errorbar(x_set, lack1means, yerr=lack1sigmas, alpha=0.5, ecolor='navajowhite', color='orange',
                         label='error')
            plt.plot(x_set, lack1means, 'orange', linewidth=4, label='predict')
            plt.plot(lack1xobs, lack1yobs, 'bo:', label='data')
            # plt.title(k, fontproperties=font)
            plt.xlabel("Date interval")
            plt.ylabel("Area ratio")
            plt.legend()
            plt.savefig(datafig_path_w + str(k) + 'lack1.jpg')
            plt.show()





df.to_excel('D:\GMM\GPR\GPtestdata.xlsx')
outex.to_excel('D:\GMM\GPR\GPtestresult.xlsx')
lack1df.to_excel('D:\GMM\GPR\GPtestdatalack1.xlsx')
lack1outex.to_excel('D:\GMM\GPR\GPtestresultlack1.xlsx')
