import  os
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv('datas/Advertising.csv')
# print(data)

#课上画园 的那个习题
# n为点的数量，n越多，越精准
def drawCirlceAndCal(n):
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
    print('圆周率是:'+str(count*4/n))
    plt.scatter(x1,y1,c=collor,s=1)
    plt.show()

#第一题的方法
def getFundationInfo():
    tv=data['TV']
    radio=data['Radio']
    newspaper=data['Newspaper']
    sales=data['Sales']
    label=['TV','Radio','Newspaper','Sales']
    datas=[tv,radio,newspaper,sales]
    for i in range(0,len(datas)):
        print(str(label[i])+'\n平均值为:'+str(datas[i].mean()))
        print('标准差为:'+str(datas[i].std()))
        print('峰度为:' + str(datas[i].kurtosis()))
        print('偏度为:' + str(datas[i].skew()))
        print('\n')
    plt.scatter(sales,tv)
    plt.show()
    plt.scatter(sales,radio)
    plt.show()
    plt.scatter(sales,newspaper)
    plt.show()

#将Sales切割成0，1序列
def divideDataInfoTwoParts():
    sales=data['Sales']
    median=sales.median()
    sales=sales.tolist()
    tv=data['TV'].tolist()
    radio=data['Radio'].tolist()
    newspaper=data['Newspaper'].tolist()
    x=[]
    y=[]
    #将y按照中位数的大小比较填入0或1
    for i in sales:
        if i>median:
            y.append(1)
        else:
            y.append(0)
    #以一行tv,radio,newspaper为一组数据，循环插入的x当中
    #时间有限，就用粗鲁的语法写了
    for i in range(0,len(tv)):
        c=[]
        c.append(tv[i])
        c.append(radio[i])
        c.append(newspaper[i])
        x.append(c)
    # for i in x:
    #     print(i)
    return x,y

#创建相应的模型
def buildDesicisonTree():
    from sklearn.datasets import load_iris
    from sklearn import tree
    clf=tree.DecisionTreeClassifier(random_state=10)
    return clf
#创建相应的模型
def buildRandomForestClassfire():
    from sklearn.ensemble import RandomForestClassifier
    clf=RandomForestClassifier(random_state=10)
    return clf
#创建相应的模型
def buildBaggin():
    from sklearn.ensemble import BaggingClassifier
    clf=BaggingClassifier(random_state=10)
    return clf

def buildLearnModel():
    x, y = divideDataInfoTwoParts()
    clf_Desicision=buildDesicisonTree()
    clf_Baggin=buildBaggin()
    clf_Random=buildRandomForestClassfire()
    xAndy=[]
    import sklearn.model_selection as sel

    # 创建决策树模型
    print('正在创建决策树模型')
    x_train, x_test, y_train, y_test = sel.train_test_split(x, y)
    clf_Desicision.fit(x_train,y_train)
    print('决策树模型创建完成')
    xy=[x_test,y_test]
    xAndy.append(xy)
    # 创建Baggin模型
    x_train, x_test, y_train, y_test = sel.train_test_split(x, y)
    print('正在创建Baggin模型')
    clf_Baggin.fit(x_train,y_train)
    print('Baggin模型创建成功')
    xy=[x_test,y_test]
    xAndy.append(xy)
    # 创建随机森林模型
    print('正在创建随机森林模型')
    x_train, x_test, y_train, y_test = sel.train_test_split(x, y)
    clf_Random.fit(x_train,y_train)
    print('森林模型创建成功')
    xy=[x_test,y_test]
    xAndy.append(xy)
    return clf_Desicision,clf_Baggin,clf_Random,xAndy


#计算acc，auc
def calAccAndAuc(models,xandy):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import roc_auc_score
    str=['决策树','Baggin','Random']
    for i in range(0,3):
        predict=models[i].predict(xandy[i][0])
        acc=accuracy_score(xandy[i][1],predict)
        yprop=models[i].predict_proba(xandy[i][0])[:, 1]
        auc=roc_auc_score(xandy[i][1],yprop)
        print(str[i])
        print(acc,auc)

#交叉对比
def compare():
    clf_Desicision, clf_Baggin, clf_Random,xAndY=buildLearnModel()
    models=[clf_Desicision, clf_Baggin, clf_Random]
    calAccAndAuc(models,xAndY)
    from sklearn.cross_validation import cross_val_score
    #调用sklearn自带的交叉对比方法
    metric=cross_val_score(clf_Desicision,xAndY[0][0],xAndY[0][1],cv=5).mean()
    print('决策树的交叉验证'+str(metric))
    metric=cross_val_score(clf_Baggin,xAndY[1][0],xAndY[1][1],cv=5).mean()
    print('Baggin的交叉验证'+str(metric))
    metric=cross_val_score(clf_Random,xAndY[2][0],xAndY[2][1],cv=5).mean()
    print('随机森林的交叉验证'+str(metric))

if __name__=='__main__':
    drawCirlceAndCal(20000)
    getFundationInfo()
    # divideDataInfoTwoParts()
    # buildLearnModel()
    compare()