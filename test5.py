from sklearn.ensemble import RandomForestClassifier

x=[[0,0],[1,1]]
y=[0,1]

# 建立随机森丽模型
clf=RandomForestClassifier(n_estimators=10)
clf=clf.fit(x,y)

# 利用随机森林模型进行预测
clf.predict_proba([2,2])
clf.predict([2,2])


# 例子2
from sklearn.datasets import load_iris
from sklearn import tree
iris=load_iris()
clf=RandomForestClassifier(n_estimators=10)
clf.fit(iris.data,iris.target)

# 利用训练好的模型机型预测
clf.predict_proba(iris.data[:1,:])
clf.predict(iris.data[:1,:])