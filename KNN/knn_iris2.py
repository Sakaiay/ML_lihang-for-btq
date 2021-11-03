# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 17:31:11 2021

@author: btq
"""
#并没有使用kd tree算法


import math
import matplotlib.pyplot as plt

def show_pic(iris_target, iris_feature, text):
    t0 = [index for index in range(len(iris_target)) if iris_target[index] == 0]
    t1 = [index for index in range(len(iris_target)) if iris_target[index] == 1]
    t2 = [index for index in range(len(iris_target)) if iris_target[index] == 2]
    
    # plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.scatter(x=iris_feature[t0, 0], y=iris_feature[t0, 1], color='r', label='Iris-virginica')
    plt.scatter(x=iris_feature[t1, 0], y=iris_feature[t1, 1], color='g', label='Iris-setosa')
    plt.scatter(x=iris_feature[t2, 0], y=iris_feature[t2, 1], color='b', label='Iris-versicolor')

    plt.xlabel("花萼长度")
    plt.ylabel("花瓣长度")
    plt.title(text)
    plt.show()

class KNNClassifier:
    def __init__(self, feature, labels):
        self.feature = feature
        self.labels = labels
    
    def get_distance(self, feature_line1, feature_line2):
        tmp = 0
        for i in range(len(feature_line1)):
            tmp += (feature_line1[i] - feature_line2[i])**2
        
        return math.sqrt(tmp)
    
    def get_type(self, k, feature_line):
        dic = {}
        for index in range(len(self.feature)):
            dist = self.get_distance(self.feature[index], feature_line)
            dic[index] = dist
        #sort
        sort_dic = sorted(dic.items(), key=lambda x:x[1], reverse=False)
        
        vote = {}
        for i in range(k):
            index = sort_dic[i][0]
            label = self.labels[index]
            if label not in vote.keys():
                vote[label] = 1
            else:
                vote[label] += 1
        vote_rank = sorted(vote.items(), key=lambda x: x[1], reverse=False)
        return vote_rank[0][0]
    def predict(self, k, feature):
        result = []
        for feature_line in feature:
            result.append(self.get_type(k, feature_line))
        return result
    
    def score(self, k, feature, labels):
        predict_set = self.predict(k, feature)
        return len([index for index in range(len(labels)) if predict_set[index] == labels[index]]) / len(labels)


from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
iris_feature = iris['data']
iris_target = iris['target']
iris_target_name = iris['target_names']

x_train, x_test, y_train, y_test = train_test_split(iris_feature, iris_target, test_size=0.33)
show_pic(iris_target, iris_feature, '原始数据')
knn = KNNClassifier(x_train, y_train)
print("测试集的准确度: {}".format(knn.score(5, x_test, y_test)))
show_pic(y_test, x_test, "测试集原始数据")

show_pic(knn.predict(5, x_test), x_test, "测试集预测数据")


