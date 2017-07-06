#coding=utf-8
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ps=np.arange(0,1,0.01)
# hs=[]
# for p in ps:
#     h=-p*np.log10(p)-(1-p)*np.log10(1-p)
#     hs.append(h)
#     print(p,"--",h)
# df=pd.DataFrame(hs,index=ps)
# df.plot()
# plt.show()

def calPi(n):
    count=0
    x1=[]
    y1=[]
    collor=[]
    for i in range(n):
        x=np.random.random()
        y=np.random.random()
        if(x**2+y**2)<1:
            count+=1
            x1.append(x)
            y1.append(y)
            collor.append('r')
        else:
            x1.append(x)
            y1.append(y)
            collor.append('g')
    print(count*4/n)
    plt.scatter(x1,y1,c=collor,s=1)
    plt.show()

def sklearnTest():
    from sklearn.datasets import load_iris
    from sklearn import tree
    iris=load_iris()
    clf=tree.DecisionTreeClassifier()
    clf=clf.fit(iris.data,iris.target)

    print(clf.predict_proba(iris.data[:1,:]))
    print(clf.predict(iris.data[:1,:]))



# calPi(1000000)
sklearnTest()