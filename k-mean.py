import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class k_mean ():
    def __init__(self, x, y, k):
        self.x = x
        self.y = y
        self.k = k
        
    def k_mean():
        
        centers = np.random.rand(k, 2) 

        max_iterations = 10 
        for it in range(max_iterations): 
            labels = np.argmin(np.sqrt(x)) 
 
            new_centers = np.array([x[labels == i].mean(axis=0) for i in range(k)]) 
    
            if np.all(centers == new_centers): 
                break
            
        centers = new_centers 
 
        clusters = [[y[labels == i] for i in range(k)]]
        
        def visulisation():
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            scatter = ax.scatter(x ,y ,c=clusters ,s=50)
            
            for i,j in clusters:
                ax.scatter(i,j,s=50,c='red',marker='+')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                
            plt.colorbar(scatter)

            fig.show()
