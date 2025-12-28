import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

class AlwaysOneClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, Y):
        return self
    
    def predict(self, X):
        return np.ones((X.shape[0],))
    
class NearestCentroidClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.centroids_ = None
        self.classes_ = None
        

    def fit(self, X, Y):
        self.classes_ = np.unique(Y)
        self.centroids_ = np.zeros((len(self.classes_), X.shape[1]))
        
        for idx, cls in enumerate(self.classes_):
            self.centroids_[idx, :] = X[Y == cls].mean(axis=0)
        
        return self
    
    def predict(self, X):
        distances = np.zeros((X.shape[0], len(self.classes_)))

        for idx, centroid in enumerate(self.centroids_):
            distances[:,idx] = np.linalg.norm(X - centroid, axis =1)
        
        return self.classes_[np.argmin(distances, axis=1)]
    
    def predict_probe(self, X):
        distances = np.zeros((X.shape[0], len(self.classes_)))

        for idx, centroid in enumerate(self.centroids_):
            distances[:, idx] = np.linalg.norm(X - centroid, axis =1)
        
        inv_distances = 1/distances
        inv_distances_sum = np.sum(inv_distances, axis=1, keepdims=True)
        return inv_distances/inv_distances_sum
    
class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.mean_ = None
        pass

    def fit(self, X, Y):
        self.mean_ = np.mean(y)
        return self
    
    def predict(self, X):
        return np.full(shape= (X.shape[0],),  fill_value=self.mean_)
    
# if __name__ == '__main__':
#     data = load_iris()
#     X, y = data.data, data.target

#     X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1)
#     clf = NearestCentroidClassifier()

#     clf.fit(X_train, y_train)

#     print(clf.score(X_test, y_test))
#     print(clf.predict(np.array([X_test[0]])))
#     print(clf.predict_probe(np.array([X_test[0]])))

if __name__ == '__main__':

#Classication Model
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1)
    clf = NearestCentroidClassifier()

    clf.fit(X_train, y_train)

    print(clf.score(X_test, y_test))
    print(clf.predict(np.array([X_test[0]])))
    print(clf.predict_probe(np.array([X_test[0]])))

    clf = KNeighborsClassifier()

    clf.fit(X_train, y_train)

    print(clf.score(X_test, y_test))

#Regression Model
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1)

    reg = MeanRegressor()
    reg.fit(X_train,y_train)

    print(reg.score(X_test, y_test))

    reg = LinearRegression()
    reg.fit(X_train,y_train)

    print(reg.score(X_test, y_test))


