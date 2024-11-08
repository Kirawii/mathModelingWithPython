import pandas as pd
import numpy as np
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
import numpy as np
from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
# myarray = pd.array([1,2,3])
# index = ['a','b','c']
# myseries = pd.Series(myarray,index = index)
# print(myseries)
# print(myseries[0])
# print(myseries['c']) 
# myarray1= np.array([[1,2,3],[2,3,4],[3,4,5]])
# rowindex = ['row1','row2','row3']
# colname = ['col1','col2','col3']
# mydataframe= pd.DataFrame(data = myarray1,index=rowindex,columns=colname)
# print(mydataframe)
# print(mydataframe['col3'])

# 数据导入
# 三种方式导入
# csv/Excel，以，分隔，文件头：字段属性
# from csv import reader
filename = '/Users/xutongkai/Downloads/f37b1ba5-97df-4627-85a4-a132692fc671.csv'
# with open(filename,'rt') as raw_data:
#     readers = reader(raw_data,delimiter= ',')
#     next(readers)
#     x = list(readers)
#     data = np.array(x).astype('float')
#     print(data.shape)

# pandas导入
from pandas import read_csv
filename = '/Users/xutongkai/Downloads/f37b1ba5-97df-4627-85a4-a132692fc671.csv'
data = read_csv(filename)
# print(data.shape)

# numpy导入
# from numpy import loadtxt
# with open (filename,'rt') as raw_data:
#     data=loadtxt(raw_data,delimiter=',')
#     print(data.shape)

# 查看数据
# head = data.head(10)
# print(head)
# print(data.dtypes)

# 查看描述性统计
# print(data.describe())

# 数据的分布
# print(data.groupby('Outcome').size())

# 数据的相关性:皮尔逊相关系数
# print(data.corr(method='pearson'))

# 数据的分布分析:高斯分布
# print(data.skew())

# 数据可视化
import matplotlib.pyplot as plt
# data.hist()
# plt.show()
# # data.plot(kind='density',subplots=True,layout=(3,3),sharex=False)
# data.plot(kind='box',subplots=True,layout=(3,3),sharex=False)
# plt.show()
columns = data.columns
# 相关矩阵
print(columns)
correlations =data.corr()
print(correlations)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations,vmin=-1,vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(columns)
ax.set_yticklabels(columns)
plt.show()

# 散点矩阵图
from pandas.plotting import scatter_matrix
scatter_matrix(data)
plt.show()
array = data.values
X =array[:,0:8]
Y =array[:,8]

transformer = MinMaxScaler(feature_range=(0,1))
newX = transformer.fit_transform(X)
set_printoptions(precision=3)
print(newX)

validation_size = 0.2 #训练集验证集八二分
seed = 7  #种子值
X_train,X_validation,Y_train,Y_validation=train_test_split(newX,Y,test_size=validation_size,random_state=seed)#分割训练集测试集
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
correlation_matrix = data.corr()

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
pyplot.scatter(X2D[:, 1], X2D[:, 1], c=y2D, edgecolor='k', s=20)
pyplot.xlabel('Feature 1')
pyplot.ylabel('Feature 2')
pyplot.title('LDA Decision Boundary')
pyplot.show()