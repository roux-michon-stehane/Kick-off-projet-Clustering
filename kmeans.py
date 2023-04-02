import numpy as np
from scipy.spatial.distance import cdist

"""
Cette implémentation du KMeans comprend 2 optimisations différentes destinées à améliorer ses performances sur des datasets de plus grande taille.

La première optimisation est l'ajout de cette variable 'tol' qui correspond au critère de tolérance, ce critère ou seuil de tolérance permet de définir la précision de convergence du modèle. En effet à chaque itération, le modèle va calculer la différence des positions des centroïdes de deux itérations successives, si cette valeur est inférieur à celle du critère de tolérance, on considère que le modèle a convergé permettant ainsi d'éviter des itérations 'inutiles'.
    
La seconde optimisation est l'ajout du kmeans++ pour l'initialisation des centroïdes. La méthode classique choisie les centroïdes de manière totalement aléatoire, ce qui peut donner de mauvais résultats. Par exemple si les centroïdes choisis se retrouvent tous être dans une zone dense d'observations celà peut conduire à une convergence prématurée sur un minimum local et fausser les résultats.
Cette initialisation kmeans++ choisie le premier centroïde de manière aléatoire. Les centroïdes suivants seront choisis avec la probabilité proportionnelle à la distance au carré de chaque observations par rapport aux centroïdes déjà choisis.
De cette manière on obtient après l'initialisation on obtient des centroïdes qui sont bien répartis dans l'espace de données et permet d'une part d'éviter la convergence vers un minimum local, et d'autre part de potentiellement réduire le nombre d'itération nécessaires pour arriver à la convergence, rendant le modèle plus rapide.
"""

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
