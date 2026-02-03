import numpy as np
def distance (X1, X2):
    return np.sum((X1 - X2)**2)

class Kmeans:
    def __init__(self, k=1):
        self.k = k
        self.X_data = None
        self.clusters = None

    def fit(self, X_data ):
        # initialize cluster, k
        choices = np.random.choice (X_data.shape[0], self.k, replace=False)
        clusters = X_data[choices]
        print (clusters)


    def predict(self, iterations=1000):
        pass
        # for number of iterations
        # for all X
        #   find distance with each cluster.
        # assign to X_i to the latest cluster
        # update the cluster to be the mean
        # find loss

if __name__ == "__main__":
    N, D = 100, 2

    kmeans = Kmeans(k=5)
    X_data = np.random.randn(N, D)
    kmeans.fit(X_data)

##  Two pointers

| K-means | link                                                                                                       |
|:-----|:--------------------------------------------------------------------------------------------------------------|
| ✅  | [ML Interview k-means](https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLC/notebooks/k_means.ipynb)             |
|  ⬜    | [Implementing K-means Clustering from Scratch - in Python](https://mmuratarat.github.io/2019-07-23/kmeans_from_scratch) |