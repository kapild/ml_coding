import numpy as np

def eucliean_distance(X1, X2):
    distance =  np.sum((X1 - X2)**2)

    distance2 = (X1 - X2).dot((X1 - X2).T)
    return distance

class KNearestNeighbor:
    def __init__(self, k=3):
        self.X_data = None
        self.y_labels = None
        self.k = k
        
    def fit(self, X_data, y_labels):
        self.X_data = X_data
        self.y_labels = y_labels

    def predict(self, X_new):

        distances  = []
        for X_i in self.X_data:
            distances.append(eucliean_distance(X_i, X_new))

        top_k = np.argsort(distances)[:self.k]
        top_labels = []
        for indx in top_k:
            top_labels.append(self.y_labels[indx])

        counts = np.bincount(top_labels)
        return np.argmax(counts)


def get_random_data(N, D):
    x_data = np.random.randn(N, D)
    y_data = np.random.randint(0, 2, (N, ))
    x_new = np.random.randn(D, 1)
    return (x_data, y_data, x_new)

if __name__ == "__main__":
    X_data = np.array([
        [0, 0],
        [1, 1],
        [1, 2],
        [2, 1],
        [0, 1],
        [1, 0],
    ])
    y_labels = [0, 1, 1, 1, 1, 1, 1]
    print ("here")
    knn = KNearestNeighbor(k=3)
    knn.fit(X_data, y_labels)

    print (knn.predict([2, 2]))

    X_data, y_labels, X_new = get_random_data(100, 4)
    knn = KNearestNeighbor(k=3)
    knn.fit(X_data, y_labels)
    print (knn.predict(X_new))
