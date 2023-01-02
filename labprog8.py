# Python program to demonstrate # KNN classification algorithm # on IRISdataset
#Write a program to implement k-Nearest Neighbour algorithm to classify the iris data set.
#Print both correct and wrong predictions. Java/Python ML library classes can be used for
#this problem.

from sklearn import datasets 
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

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
