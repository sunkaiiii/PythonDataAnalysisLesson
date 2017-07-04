#coding=utf-8
import os
import pandas as pd
import matplotlib.pyplot as plt

# help(pd.read_excel)
files=os.listdir('datas/课堂作业')
os.chdir('datas/课堂作业')
for i in files:
    df=pd.read_excel(i,skiprows=[0,1],skip_footer=2,index_col=0,header=None,name='利率')
    print(df)
    df.resample('M').mean().plot()
    plt.show()