from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLarsIC
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

def question1():
    diabetes=datasets.load_diabetes()

    X = diabetes.data
    y = diabetes.target

    X = pd.DataFrame(X, columns=['x' + str(i) for i in range(X.shape[1])])
    Y = pd.DataFrame(y, columns=['Y'])
    data=pd.concat([Y,X],axis=1)
    data.to_csv("day4_project_data.csv")
    # print(data)
    # 将测试集按照3:1来分配训练集和测试集
    x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=10,test_size=0.25)
    return x_train,x_test,y_train,y_test

def question2():
    #做线性回归模型，代码比较好理解
    x_train,x_test,y_train,y_test=question1()
    linear_regression=LinearRegression()
    linear_regression.fit(x_train,y_train)
    y_pred = linear_regression.predict(x_test)
    mse = np.mean((y_test - y_pred) ** 2)
    print('线性回归模型的mse是:'+str(mse.tolist()))

def question3():
    #同样比较好理解
    x_train, x_test, y_train, y_test = question1()
    alphas = np.logspace(-10, 10, 20)
    Ridge1 = RidgeCV(alphas=alphas)
    Ridge1 = Ridge1.fit(x_train, y_train)
    theBestAlpha = Ridge1.alpha_

    Ridge2 = Ridge(alpha=theBestAlpha)
    Ridge2 = Ridge2.fit(x_train, y_train)
    y_pred = Ridge2.predict(x_test)
    mse2 = np.mean((y_test-y_pred)**2)
    print('岭回归模型的mse是:' + str(mse2.tolist()))

    #画出岭回归图
    coef=[]
    for i in alphas:
        Ridge2.set_params(alpha=i)
        Ridge2.fit(x_train,y_train)
        coef.append(Ridge2.coef_)
    draw=plt.gca()
    draw.set_xscale('log')
    draw.plot(alphas,coef)
    draw.set_xlim(draw.get_xlim()[::-1])
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.show()


def question4():
    x_train, x_test, y_train, y_test = question1()
    alphas = np.logspace(-10, 10, 20)
    lasso=LassoCV(alphas=alphas)
    lasso=lasso.fit(x_train,y_train)
    theBestAlpha = lasso.alpha_

    lasso2=Lasso(alpha=theBestAlpha)
    lasso2=lasso2.fit(x_train,y_train)
    y_pred=lasso2.predict(x_test)
    mse3=np.mean((y_test-y_pred)**2)
    print('Lasso的mse是'+str(mse3))

    #根据官方文档（criterion : ‘bic’ | ‘aic’The type of criterion to use.）criterion参数要标注才会有效果
    crossmodel=LassoLarsIC(criterion='aic')
    crossmodel.fit(x_test,y_test)
    crossmodel2=LassoLarsIC(criterion='bic')
    crossmodel2.fit(x_test,y_test)
    critertion=crossmodel.criterion_
    alphas=crossmodel.alphas_
    # 画图
    plt.plot(-np.log10(alphas),critertion,color='r',label='croomodel1')
    critertion2 = crossmodel2.criterion_
    alphas2 = crossmodel2.alphas_
    plt.plot(-np.log10(alphas2), critertion2, color='g', label='croomodel2')
    plt.xlabel('-log(alpha)')
    plt.ylabel('critertion')
    plt.show()

# question1()
question2()
question3()
question4()