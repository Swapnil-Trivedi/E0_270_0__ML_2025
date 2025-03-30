import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import GridSearchCV

class GridSearch:
    def __init__(self,cv=5):
        self.cv = cv

    def tune_knn(self,x_train,y_train):       
        param_grid = {
        'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],  # Possible values for k (number of neighbors)
        'p': [1, 2,3,4,5,6,7,8,9,10],  # p = 1 for Manhattan distance, p = 2 for Euclidean distance
        'weights': ['uniform', 'distance']  # 'uniform' for equal weight, 'distance' for distance-based weight
        }
        knn=KNeighborsClassifier()
        grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=self.cv, n_jobs=-1, verbose=1)
        grid_search.fit(x_train, y_train)

        # Get the best model and parameters
        best_params = grid_search.best_params_

        return grid_search.best_score_, best_params , 

    def _plot_knn_results(self, df_results):
       #plot k vs accuracy
        plt.plot(df_results['K'], df_results['Accuracy'])
        plt.xlabel('K')
        plt.ylabel('Accuracy')
        plt.title('KNN Performance')
        plt.show()
    
    def tune_logistic_regression(self,x_train,y_train):
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'],'solver':['liblinear','saga'],'max_iter':[500]}
        log_reg = LogisticRegression()
        grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=self.cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)
        print("Best Parameters:", grid_search.best_params_)
        print("Best Cross-Val Score:", grid_search.best_score_)
        return grid_search.best_params_,grid_search.best_score_

    def tune_naive_bayes_classifier(self,x_train,y_train):
        model=GaussianNB()
        param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5,1e-4,1e-3,1e-2]}
        grid_search = GridSearchCV(model, param_grid, cv=self.cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)        
        print("Best parameters:", grid_search.best_params_)
        return grid_search.best_params_
    def tune_linear_svm(self,model,x_train,y_train):
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100]  # Different values for regularization
        }

        # Grid Search with Cross-Validation
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)

        # Best Parameters & Score
        print("Best Parameters:", grid_search.best_params_)
        print("Best Validation Accuracy:", grid_search.best_score_)
        return grid_search.best_params_
    
    def tune_rbf_svm(self,model,x_train,y_train):
        param_grid = {
            'C': [0.1, 1, 10, 100],  # Regularization
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]  # RBF kernel gamma values
        }

        grid_search = GridSearchCV(model, param_grid, cv=self.cv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)

        # Best params
        print("Best Parameters:", grid_search.best_params_)
        print("Best Validation Accuracy:", grid_search.best_score_)


