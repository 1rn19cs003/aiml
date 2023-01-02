#Apply EM algorithm to cluster a set of data stored in a .CSV file. Use the same data set
#for clustering using k-Means algorithm. Compare the results of these two algorithms and
#comment on the quality of clustering. You can add Java/Python ML library classes/API in
#the program.

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np


iris = datasets.load_iris() 
print(iris)
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target) 
model =KMeans(n_clusters=3)
model.fit(X_train,y_train) 
model.score
print('K-Mean: ',metrics.accuracy_score(y_test,model.predict(X_test)))

#-------Expectation and Maximization----------
from sklearn.mixture import GaussianMixture 
model2 = GaussianMixture(n_components=3) 
model2.fit(X_train,y_train)
model2.score
print('EM Algorithm:',metrics.accuracy_score(y_test,model2.predict(X_test)))
