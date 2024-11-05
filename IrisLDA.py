from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np

filename = '/Users/xutongkai/Downloads/iris (2).csv'
names =['Sepal-Length','Sepal-Width','Petal-Length','Petal-Width','Species']
dataset = read_csv(filename,skiprows=1,names=names)#读取文件，skiprows=1是跳过第一行，因为第一行是列名，属于object类型，后续不能调用pandas计算，自定义names来赋予列名
cols_to_plot = dataset.columns[:-1]#最后一列用0，1，2标注了species，在无明显数量差异的情况下，不用进行分析
dataset.hist()#绘制直方图
pyplot.show()
scatter_matrix(dataset[cols_to_plot], alpha=0.2, figsize=(6, 6), diagonal='kde')#绘制散点图矩阵，四个参数分别为数据集、点的不透明度，图形大小、diagonal='kde' 指定了对角线上显示核密度估计图。
pyplot.show()
array =dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.2 #训练集验证集八二分
seed = 7  #种子值
X_train,X_validation,Y_train,Y_validation=train_test_split(X,Y,test_size=validation_size,random_state=seed)#分割训练集测试集
#调用不同模型测试
models = {} 
models['LR'] = LogisticRegression()
models['KNN']=KNeighborsClassifier()
models['LDA']=LinearDiscriminantAnalysis()
models['CART']=DecisionTreeClassifier()
models['SVM']=SVC()
models['NB']=GaussianNB()

results = []
for key in models:
    Kfold = KFold(n_splits=10,shuffle=True,random_state=seed)#十次交叉验证
    cv_results = cross_val_score(models[key],X_train,Y_train,cv=Kfold,scoring='accuracy')#评估模型性能，模型、训练集 cv=kfold表示调用之前创建的KFold交叉验证对象，accuracy表示用准确性来评估
    results.append(cv_results)
    print('%s:%f(%f)'%(key,cv_results.mean(),cv_results.std()))#输出平均值和标准差
lda=LinearDiscriminantAnalysis()#选取最优模型进一步分析
lda.fit(X=X_train,y=Y_train)#训练模型
predictions = lda.predict(X_validation)#预测集
print(accuracy_score(Y_validation,predictions))#准确性得分
print(confusion_matrix(Y_validation,predictions))#混淆矩阵
print(classification_report(Y_validation,predictions))#显示主要的分类指标，如精确度、召回率、F1分数等

# 获取LDA系数
coefficients = lda.coef_

# 绘制系数
pyplot.bar(range(len(coefficients[0])), coefficients[0])
pyplot.xlabel('Feature Index')
pyplot.ylabel('Coefficient Value')
pyplot.title('LDA Coefficients')
pyplot.show()

import seaborn as sns

# 计算相关性矩阵
correlation_matrix = dataset.corr()

# 可视化相关性矩阵
pyplot.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
pyplot.title('Correlation Matrix')
pyplot.show()


# 选择两个特征进行绘图
X2D = X_train[:, [2, 3]]
y2D = Y_train

# 训练LDA模型
lda = LinearDiscriminantAnalysis()
lda.fit(X2D, y2D)

# 创建网格来评估模型
x_min, x_max = X2D[:, 0].min() - 1, X2D[:, 0].max() + 1
y_min, y_max = X2D[:, 1].min() - 1, X2D[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# 预测网格点的类别
Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界和训练点
pyplot.contourf(xx, yy, Z, alpha=0.4)
pyplot.scatter(X2D[:, 0], X2D[:, 1], c=y2D, edgecolor='k', s=20)
pyplot.xlabel('Feature 1')
pyplot.ylabel('Feature 2')
pyplot.title('LDA Decision Boundary')
pyplot.show()