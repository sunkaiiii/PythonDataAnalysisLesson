#coding=utf-8
##课堂练习

import os
import  pandas as pd
import numpy as np

def lessonTest():
    sh1=pd.read_excel('datas/index/上证综指.xlsx')
    rm=sh1.pctchg

    data1=pd.read_excel('datas/stock/000001.xlsx')
    ri=data1.pctchg

    x=np.asanyarray(ri).reshape(4234,1)
    y=np.asanyarray(rm).reshape(4234,1)

    from sklearn.linear_model import LinearRegression
    lm1=LinearRegression()
    lm1=lm1.fit(x,y)

    beta=lm1.coef_
    print(beta)

lessonTest()
