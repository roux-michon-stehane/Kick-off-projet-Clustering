import numpy as np
from scipy.spatial.distance import cdist

class myKMeans:
    def __init__(self, k=8, init='random', max_iter=300, tol=1e-4):
        self.k = k
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None

    def fit(self, X):
        n_samples, n_features = X.shape
        if self.init == 'random':
            self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]

        elif self.init == 'kmeans++':
            self.centroids = [X[np.random.choice(n_samples)]]
            while len(self.centroids) < self.k:
                dists = cdist(X, self.centroids)
                probs = np.min(dists, axis=1) ** 2
                probs /= np.sum(probs)
                idx = np.random.choice(n_samples, p=probs)
                self.centroids.append(X[idx])
            self.centroids = np.array(self.centroids)

        for _ in range(self.max_iter):
            distances = cdist(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)
            new_centroids = np.zeros((self.k, n_features))
            for i in range(self.k):
                new_centroids[i] = X[self.labels == i].mean(axis=0)
            if np.abs(new_centroids - self.centroids).sum() < self.tol:
                break

            self.centroids = new_centroids

    def predict(self, X):
        distances = cdist(X, self.centroids)
        return np.argmin(distances, axis=1)
