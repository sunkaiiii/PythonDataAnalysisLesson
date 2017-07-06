import numpy as np
import matplotlib.pyplot as plt
from sklearn import  linear_model
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Ridge,RidgeCV
data=pd.read_csv('datas/Advertising.csv')
def test1():
    x=1./(np.arange(1,11)+np.arange(0,10)[:,np.newaxis])
    y=np.ones(10)

    n_alphas=200
    alphas=np.logspace(-10,-2,n_alphas)
    clf=linear_model.Ridge(fit_intercept=False)

    conefs=[]
    for a in alphas:
        clf.set_params(alpha=a)
        clf.fit(x,y)
        conefs.append(clf.coef_)

    ax=plt.gca()

    ax.plot(alphas,conefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients')
    plt.axis('tights')
    plt.show()

def test2():
    reg=linear_model.Lasso(alpha=0.1)
    reg.fit([[0,0],[1,1]],[0,1])
    reg.coef_
    print(reg.predict([[1,1]]))


def test3():



    x=data[['TV','Radio','Newspaper']]
    y=data['Sales']
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=10)


    lm1=LinearRegression()
    lm1=lm1.fit(x_train,y_train)
    print(lm1.coef_)
    print(lm1.intercept_)
    print(lm1.score(x_train,y_train))

    y_pred=lm1.predict(x_test)

    mse1=np.mean((y_test-y_pred)**2)
    MSE=[]
    MSE.append(mse1)


    alphas=np.logspace(-10,10,20)
    Ridge1=RidgeCV(alphas=alphas)
    Ridge1=Ridge1.fit(x_train,y_train)
    print(Ridge1.alpha_)
    theBestAlpha=Ridge1.alpha_

    Ridge2=Ridge(alpha=theBestAlpha)
    Ridge2=Ridge2.fit(x_train,y_train)

    print(Ridge2.coef_)
    print(Ridge2.intercept_)


    y_pred=Ridge2.predict(x_test)
    MSE2=np.mean((y_pred-y_test))
    print(MSE2)
    MSE.append(MSE2)

test3()