#coding=utf-8
#!/usr/bin/python3.5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# s=pd.Series([1,3,5,np.nan,6,8])
# print(s)
#
# s2=pd.Series([1,3,5,6,8],index=['a','b','c','d','e'])
# print(s2)


# s3=pd.Series(np.random.random(1000),index=pd.date_range('2000-01-01',periods=1000))
# print(s3)

# dic={"sunkai":1,"wangyichun":2,"zhangshuai":3}
# s4=pd.Series(dic)
# print(s4)

# stata={'Ohin':35,'Texas':71,'Oregon':16,'Utah':50}
# State=['Califunoa','Uatah','Oregon','Texas']
# s5=pd.Series(stata,index=State)
# s5.name='population'
# s5.index.name='state'
# print(s5)

# detal=pd.date_range('2001-01-01',periods=10)
# print(detal)
#
# dates=pd.date_range('20130101',periods=6)
# df=pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('date'))
# print(df)
# df.plot()
# plt.show()
#

def comintegralbyladder(func,x0,x1):
    wholearea=0
    step=0.1
    for i in np.arange(x0,x1,step):
        wholearea+=(func(i)+func(i+step))*step/2
    return wholearea

print('1',comintegralbyladder(np.exp,1,4))

def comintegralbyladder2(func,x0,x1,n):
    wholearea=0
    step=(x1-x0)/n
    for i in np.arange(x0,x1,step):
        wholearea+=(func(i)+func(i+step))*step/2
    return wholearea

print('2',comintegralbyladder2(np.exp,1,4,300))


print('正确',np.exp(4)-np.exp(1))


x=np.linspace(-5,5,num=100)
y=np.exp(x)
plt.plot(x,y)


list1=[]
for i in range(10,100000,100):
    temp=comintegralbyladder2(np.exp,1,4,i)
    error=np.exp(4)-np.exp(1)-temp
    print(i,error)
    list1.append(error)
a=range(10,100000,100)
a=list(a)
df=pd.DataFrame(list1,index=a)
df.plot()
plt.show()

