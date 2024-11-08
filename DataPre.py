import numpy as np
from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
filename = '/Users/xutongkai/Downloads/f37b1ba5-97df-4627-85a4-a132692fc671.csv'
data = read_csv(filename)
array = data.values
# 调整数据尺度--数据缩放
X =array[1:,0:8]
Y =array[:,8]
transformer = MinMaxScaler(feature_range=(0,1))
newX = transformer.fit_transform(X)
set_printoptions(precision=3)
print(newX)

#正态化数据
transformer = StandardScaler().fit(X)
newX = transformer.transform(X)
set_printoptions(precision=3)
print(newX)

# 标准化数据，归一化

transformer =Normalizer().fit(X)
newX = transformer.transform(X)
set_printoptions(precision=3)
print(newX)

#二值处理

transformer =Binarizer(threshold=0.0).fit(X)
newX = transformer.transform(X)
set_printoptions(precision=3)
print(newX)


