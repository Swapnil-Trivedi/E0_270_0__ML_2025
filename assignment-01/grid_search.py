import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV

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
    
    def tune_logistic_regression(self,model,x_train,y_train):
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'],'solver':['liblinear','saga'],'max_iter':[700]}
        log_reg = model
        grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)
        print("Best Parameters:", grid_search.best_params_)
        print("Best Cross-Val Score:", grid_search.best_score_)
        return grid_search.best_params_,grid_search.best_score_

    def tune_naive_bayes_classifier(self,model,x_train,y_train):
        param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
        grid_search = GridSearchCV(model, param_grid, cv=self.cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)        
        print("Best parameters:", grid_search.best_params_)
        return grid_search.best_params_


