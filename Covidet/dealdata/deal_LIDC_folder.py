import os
import shutil
def findDepestFolder(path,finallist,deepnum = 0):
    dir = os.listdir(path)
    if len(dir) == 0:
        return
    dirs = []
    for ForD in dir:
        if os.path.isdir(os.path.join(path,ForD)):
            dirs.append(os.path.join(path,ForD))
    if len(dirs) == 0:
        finallist.append(path)
    else:
        for d in dirs:
            dd = os.path.join(path,d)
            findDepestFolder(dd,finallist,deepnum+1)
path = '/data/chenxiangru/covidData/neg_no_pneumina'
folderlist = []
findDepestFolder(path,folderlist)
for d in folderlist:
    lenth = len(os.listdir(d))
    if lenth <5:
        shutil.rmtree(d)