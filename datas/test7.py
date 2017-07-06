# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 22:33:08 2017

@author: GSY
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

X = pd.DataFrame(X, columns=['x' + str(i) for i in range(X.shape[1])])
Y = pd.DataFrame(y, columns=['Y'])

data = pd.concat([Y, X], axis=1)
data.to_csv("day4_project_data.csv")
MSE = []
##Step1

from sklearn.cross_validation import train_test_split

# 分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10,test_size=0.25)

# ##Step2  OLS
# from sklearn.linear_model import LinearRegression
#
# lm1 = LinearRegression()
# lm1 = lm1.fit(X_train, y_train)
# lm1.coef_
# lm1.intercept_
# lm1.score(X_train, y_train)
#
# y_pred = lm1.predict(X_test)
# # 求相关系数
# MSE1 = np.mean((y_test - y_pred) ** 2)
# MSE.append(MSE1)
#
# ##Step3 Ridge
# from sklearn.linear_model import Ridge, RidgeCV
#
# alphas = np.logspace(-10, 10, 20)
# Ridge1 = RidgeCV(alphas=alphas)
# Ridge1 = Ridge1.fit(X_train, y_train)
# TheBestAlpha = Ridge1.alpha_
#
# Ridge2 = Ridge(alpha=TheBestAlpha)
# Ridge2 = Ridge2.fit(X_train, y_train)
# Ridge2.coef_
# Ridge2.intercept_
#
# y_pred = Ridge2.predict(X_test)
# MSE2 = np.mean((y_pred - y_test) ** 2)
#
# MSE.append(MSE2)
#
# # 画岭迹图
# coefs = []
# for a in alphas:
#     Ridge2.set_params(alpha=a)  # 遍历alpha
#     Ridge2.fit(X, y)  # 分别计算系数值
#     coefs.append(Ridge2.coef_)
#
# ax = plt.gca()  # 岭迹图
#
# ax.plot(alphas, coefs)
# ax.set_xscale('log')
# ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
# plt.xlabel('alpha')
# plt.ylabel('weights')
# plt.title('Ridge coefficients as a function of the regularization')
# plt.axis('tight')
# plt.show()

##Step4 Lasso
from sklearn.linear_model import Lasso, LassoCV, LassoLarsIC
from sklearn.metrics import r2_score
import time

alphas = np.logspace(-10, 10, 20)
Lasso1 = LassoCV(alphas=alphas)
Lasso1 = Lasso1.fit(X_train, y_train)
TheBestAlpha = Lasso1.alpha_

Lasso2 = Lasso(alpha=TheBestAlpha)
Lasso2 = Lasso2.fit(X_train, y_train)
Lasso2.coef_
Lasso2.intercept_

y_pred = Lasso2.predict(X_test)
MSE3 = np.mean((y_pred - y_test) ** 2)
print(MSE3)

MSE.append(MSE3)

# 画Lasso模型分析图
model_bic = LassoLarsIC(criterion='bic')
t1 = time.time()
model_bic.fit(X_test, y_test)
t_bic = time.time() - t1
alpha_bic_ = model_bic.alpha_

model_aic = LassoLarsIC(criterion='aic')
model_aic.fit(X_test, y_test)
alpha_aic_ = model_aic.alpha_


def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    criterion_ = model.criterion_
    plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
             linewidth=3, label='%s criterion' % name)
    plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
                label='alpha: %s estimate' % name)
    plt.xlabel('-log(alpha)')
    plt.ylabel('criterion')


plt.figure()
plot_ic_criterion(model_aic, 'AIC', 'b')
plot_ic_criterion(model_bic, 'BIC', 'r')
plt.legend()
plt.title('Information-criterion for model selection (training time %.3fs)'
          % t_bic)
plt.show()