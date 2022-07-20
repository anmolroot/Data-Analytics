# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 09:41:43 2022

@author: Student
"""

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


dataset = load_iris()

x = dataset.data
y = dataset.target
 

plt.scatter(x[y == 0,0], x[y == 0,1],color = 'red', label='setosa')
plt.scatter(x[y == 1,0], x[y == 1,1],color = 'green', label='versicolor' )
plt.scatter(x[y == 2,0], x[y == 2,1],color = 'blue', label='verginica' )
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.legend()
plt.title("analysis on iris dataset")
plt.show()

""" Insight : sepal lenght and sepal width are not good predictors
as they cannot differentiate b/w versicolor and verginica
"""

plt.scatter(x[y == 0,2], x[y == 0,3],color = 'red', label='setosa')
plt.scatter(x[y == 1,2], x[y == 1,3],color = 'green', label='versicolor' )
plt.scatter(x[y == 2,2], x[y == 2,3],color = 'blue', label='verginica' )
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.legend()
plt.title("analysis on iris dataset")
plt.show()

""" Insight : petal lenght and petal width are good predictors
as they can differentiate b/w versicolor and verginica and setosa
"""