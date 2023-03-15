import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
df_iris = pd.DataFrame(iris.data)
df_iris.columns=["Longueur_sepale", "Largeur_sepale", "Longueur_petale", "Largeur_petale"]
df.head()

class k_mean ():
    def __init__(self, X_, k):
        self.X_ = x
        self.k = k
      
    def converteur ():
        x_min, x_max = X_[:, 0].min() - 0.5, X_[:, 0].max() + 0.5
        y_min, y_max = X_[:, 1].min() - 0.5, X_[:, 1].max() + 0.5

    def k_mean():
        
        centers = np.random.rand(k, 2) 

        max_iterations = 10 
        for it in range(max_iterations): 
            labels = np.argmin(np.sqrt(X_[:,1])) 
 
            new_centers = np.array([X_[:,0].mean(axis=0)]) 
    
            if np.all(centers == new_centers): 
                break 
            
        centers = new_centers 
 
        clusters = [[[centers] for i in range(k)]]
        
        def visulisation():
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            scatter = ax.scatter(X_[:, 0], X_[:, 1] ,c= clusters ,s=50)
            for i,j in clusters:
                ax.scatter(i,j,s=50,c='red',marker='+')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.colorbar(scatter)

            fig.show()
            return()
