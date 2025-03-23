from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import accuracy_score
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

    def tune_logistic_regression(self,model,x_train,y_train):
        param_space = {
            'C': Real(0.001, 100, prior='log-uniform'),  # Regularization strength
            'penalty': Categorical(['l1', 'l2']),  # Regularization type
            'solver': Categorical(['liblinear', 'saga']),  # Optimization algorithm
            'max_iter': Integer(100, 500)  # Maximum iterations
        }

        # Initialize Bayesian Search with Logistic Regression
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=param_space,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring='accuracy',
            n_jobs=-1,  # Use all available CPUs
            random_state=42
        )

        # Fit the model
        bayes_search.fit(x_train, y_train)
        print("Best Params:", bayes_search.best_params_)
        print("Best Validation Accuracy:", bayes_search.best_score_)
        # Return best parameters and accuracy
        return bayes_search.best_params_, bayes_search.best_score_

    def tune_naive_bayes_classifier(self,model,x_train,y_train):
        param_space = {
                'var_smoothing': Real(1e-9, 1e-2, prior='log-uniform')
            }

        # Initialize Bayesian Optimization
        bayes_search = BayesSearchCV(
            model, 
            param_space, 
            n_iter=20,  # Number of iterations
            cv=5,       # 5-fold cross-validation
            scoring='accuracy', 
            n_jobs=-1
        )
        # Fit the model
        bayes_search.fit(x_train, y_train)
        # Best hyperparameter
        print("Best Params:", bayes_search.best_params_)
        print("Best Validation Accuracy:", bayes_search.best_score_)
        return bayes_search.best_estimator_
        
        
              