import numpy as np
def distance (X1, X2):
    return np.sum((X1 - X2)**2)

class Kmeans:
    def __init__(self, k=1):
        self.k = k
        self.X_data = None
        self.clusters = None

    def fit(self, X_data ):
        self.X_data = X_data
        centroids = np.random.choice(self.X_data.shape[0], self.k, replace=False)
        self.clusters = X_data[centroids]

    def predict(self, iterations=1000):

if __name__ == "__main__":
    N, D = 100, 2

    kmeans = Kmeans(k=5)
    X_data = np.random.randn(N, D)
    kmeans.fit(X_data)
