import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class k_mean ():
    def __init__(self, x_init, K):
        self.X_ = X_
        self.K = K
      
    def converteur (self):
        x_min, x_max = X_[:, 0].min() - 0.5, X_[:, 0].max() + 0.5
        y_min, y_max = X_[:, 1].min() - 0.5, X_[:, 1].max() + 0.5

        return self.X_
        
    # Initialize les centroide de l'agoritme k-mean
    def init_centroids(X, K):
        m, n = X_.shape
        centroids = np.zeros((k, 2))
        for i in range(k):
            centroid = X_[np.random.choice(range(m))]
            centroids[i] = centroid
        print(centroids)
        return centroids

    # Calculate the distance between each data point and each centroid
    def distance(X, k, centroids):
        k = centroids.shape[0]
        dist = np.zeros((X_.shape[0], k))
        for i in range(k):
            dist[:, i] = np.linalg.norm(X_ - centroids[i], axis=1)
        print(dist)
        return dist

    # Assign each data point to the closest centroid
    def assign_centroids(X, dist):
        return np.argmin(dist, axis=1)

    # Calculate the new centroids based on the mean of each cluster
    def calculate_centroids(X, idx, K):
        n = X_.shape[1]
        centroids = np.zeros((k, n))
        for i in range(k):
            centroids[i, :] = np.mean(X_, axis=0)
        return centroids
    
    # Implementation de la visualisation
    def visualization (k, X_, centroids):
        for j in range(k):
            centroids[j] = np.mean(X_[assignments == j], axis=0)

    # créer un objet Figure et un objet Axes
        colors = ['b', 'g', 'r']
        for j in range(k):
            mask = (assignments == j)
            plt.scatter(X_[mask, 0], X_[mask, 1], c=colors[j], label='Cluster {}'.format(j+1))
            plt.scatter(centroids[j, 0], centroids[j, 1], c='k', marker='x')
        
    # ajouter des libellés et un titre    
        plt.xlabel('premier partie de la donnes')
        plt.ylabel('deuxieme partie de la donnes')
        plt.title(' la visualisation du model de clusteringK-means')
        plt.legend()
        plt.show()    

    # Implementation du K-means clustering
    def kmeans(X, K, max_iters,self):
        centroids = init_centroids(X, K)
        for i in range(max_iters):
            dist = distance(X, centroids)
            idx = assign_centroids(X, dist)
            centroids = calculate_centroids(X, idx, K)
        self.visualization   
        return idx, centroids,visualization 
