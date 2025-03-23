from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from skopt import BayesSearchCV
import pandas as pd


class BayesianSearch:
    def __init__(self, cv=5, n_iter=30):
        self.cv = cv
        self.n_iter = n_iter  # Number of iterations for Bayesian optimization
    
    def tune_knn(self, x_train, y_train):
        search_space = {
            'n_neighbors': (1, 30)  # Only tuning K
        }
        
        knn = KNeighborsClassifier()
        bayes_search = BayesSearchCV(
            knn, 
            search_spaces=search_space,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )

        bayes_search.fit(x_train, y_train)

        # Store results
        df_results = pd.DataFrame({
            'Iteration': range(1, len(bayes_search.cv_results_['param_n_neighbors']) + 1),
            'K': bayes_search.cv_results_['param_n_neighbors'].data,
            'Accuracy': bayes_search.cv_results_['mean_test_score']
        })
        
        best_k = bayes_search.best_params_['n_neighbors']
        best_score = bayes_search.best_score_

        return best_k, best_score
