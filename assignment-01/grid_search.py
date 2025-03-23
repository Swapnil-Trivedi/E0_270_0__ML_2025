import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

class GridSearch:
    def __init__(self,cv=5):
        self.cv = cv

    def tune_knn(self,x_train,y_train,plot=True):       
        k_range = range(1, 30)
        results = []
        max_score = 0
        best_k = None
        for k in k_range:
            print(f"Trying K={k}")
            knn = KNeighborsClassifier(n_neighbors=k)
            score = cross_val_score(knn,x_train,y_train, cv=self.cv, scoring='accuracy').mean()
            results.append({'K': k, 'Accuracy': score})
            #store max accuracy and best hyperparameters
            if score > max_score:
                max_score = score
                best_k = k
                    
        df_results = pd.DataFrame(results)

        #Plot results
        if plot:
            self._plot_knn_results(df_results)

        return (best_k, max_score)

    def _plot_knn_results(self, df_results):
       #plot k vs accuracy
        plt.plot(df_results['K'], df_results['Accuracy'])
        plt.xlabel('K')
        plt.ylabel('Accuracy')
        plt.title('KNN Performance')
        plt.show()


