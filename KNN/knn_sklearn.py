# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:14:06 2021

@author: btq
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
# =============================================================================
# 1.5 40 thin
# 1.5 45 thin
# 1.5 50 fat
# 1.5 60 fat
# 1.6 40 thin
# 1.6 50 thin
# 1.6 60 fat
# 1.6 70 fat
# 1.7 50 thin
# 1.7 60 thin
# 1.7 70 fat
# 1.7 80 fat
# 1.8 60 thin
# 1.8 70 thin
# 1.8 80 fat
# 1.8 90 fat
# 1.9 80 thin
# 1.9 85 thin
# 1.9 100 fat
# 1.9 110 fat
# =============================================================================
knn_data = pd.read_excel('D:/python_demo/knn_data.xlsx',header=None)
label = knn_data[2]
hight = knn_data[0]
weight = knn_data[1]
data = [[hight[i],weight[i]] for i in range(len(hight))]

x = np.array(data)
label = np.array(label)
y = np.zeros(label.shape)
y[label=='fat']=1

''' 训练KNN分类器 '''
clf = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors = 3)
clf.fit(x, y)

'''测试结果的打印'''
answer = clf.predict(x)
print(x)
print(answer)
print(y)
print(np.mean( answer == y))


precision, recall, thresholds = precision_recall_curve(y, clf.predict(x))
answer = clf.predict(x)
print(classification_report(y, answer, target_names = ['thin', 'fat']))


plt.rcParams['font.sans-serif'] = ['KaiTi'] 
plt.rcParams['axes.unicode_minus'] = False
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel(u'身高')
plt.ylabel(u'体重')
plt.show()
