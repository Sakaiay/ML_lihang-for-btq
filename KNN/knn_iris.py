# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:56:21 2021

@author: btq
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
iris_data = iris.data
iris_target = iris.target
X_train, X_test, Y_train, Y_test= train_test_split(iris_data, iris_target, test_size=0.2, random_state=0)

#model train
# select kd tree algorithm
model = KNeighborsClassifier(algorithm='kd_tree', n_neighbors = 5)
model.fit(X_train, Y_train)

#model predict
Y_pred = model.predict(X_test)

#model evaluation
correct_pred = np.count_nonzero(Y_pred==Y_test)
accuracy = correct_pred / len(Y_test)
print("model accuracy is " + str(accuracy))


print("model accuracy is {}".format(accuracy_score(Y_test, Y_pred)))