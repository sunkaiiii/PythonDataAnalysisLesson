#coding=utf-8
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def getAvgMaxMultiple():
    print('正稍等，正在处理中')
    files=os.listdir('datas/day2-data')
    os.chdir('datas/day2-data')
    dates=[]
    value=[]
    for i in files:
        df = pd.read_excel(i)
        datas=df[df['债券类型'] == "国债"]
        meanPrince=datas["理论最大放大倍数"].mean()
        dates.append(datetime.datetime.strptime(i.split('.')[0][:-3],'%Y-%m'))
        value.append(meanPrince)

    # print(dates)
    plt.plot(dates,value)
    plt.show()

if __name__=='__main__':
    getAvgMaxMultiple()