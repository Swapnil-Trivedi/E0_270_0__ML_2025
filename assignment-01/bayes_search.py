from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import pandas as pd


class BayesianSearch:
    def __init__(self, cv=3, n_iter=30):
        self.cv = cv
        self.n_iter = n_iter  # Number of iterations for Bayesian optimization
    
    def tune_knn(self, x_train, y_train):
        search_space = {
            'n_neighbors': (1, 30),  #tuning K
            'p' : (1,10),
            'weights' : ['uniform','distance']
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

        return best_score , bayes_search.best_params_

    def tune_logistic_regression(self,x_train,y_train):
        param_space = {
            'C': Real(0.001, 100, prior='log-uniform'),  # Regularization strength
            'penalty': Categorical(['l1', 'l2']),  # Regularization type
            'solver': Categorical(['liblinear', 'saga']),  # Optimization algorithm
            'max_iter': Integer(100, 500)  # Maximum iterations
        }
        model = LogisticRegression()
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
            n_iter=self.n_iter,  # Number of iterations
            cv=self.c,
            scoring='accuracy', 
            n_jobs=-1
        )
        # Fit the model
        bayes_search.fit(x_train, y_train)
        # Best hyperparameter
        print("Best Params:", bayes_search.best_params_)
        print("Best Validation Accuracy:", bayes_search.best_score_)
        return bayes_search.best_estimator_
    
    def tune_linear_svm(self,model,x_train,y_train):
        param_space = {
            'C': Real(0.0001, 100, prior='log-uniform')  # C in a log-uniform scale
        }
        
        # Initialize Bayesian Search
        bayes_search = BayesSearchCV(model, param_space, n_iter=self.n_iter, cv=self.cv, scoring='accuracy', n_jobs=-1)
        bayes_search.fit(x_train, y_train)
        
        # Best Parameters & Score
        print("Best Parameters:", bayes_search.best_params_)
        print("Best Validation Accuracy:", bayes_search.best_score_)
        return bayes_search.best_params_
    
    def tune_rbf_svm(self,model,x_train,y_train):
        param_space_numeric = {
            'C': Real(0.0001, 1, prior='log-uniform'),  # C in a log-uniform scale
            'gamma': Real(0.01,1,prior="log-uniform")
        }
        
        # Initialize Bayesian Search
        bayes_search_numeric = BayesSearchCV(model, param_space_numeric, n_iter=self.n_iter, cv=self.cv, scoring='accuracy', n_jobs=-1)
        bayes_search_numeric.fit(x_train, y_train)
        
        param_space_categorical = {
             'C': Real(0.0001, 1, prior='log-uniform'),
             'gamma': Categorical(["scale", "auto"])
            }
        bayes_search_categorical = BayesSearchCV(model, param_space_categorical, n_iter=self.n_iter, cv=self.cv, scoring='accuracy', n_jobs=-1)
        bayes_search_categorical.fit(x_train, y_train)

        if bayes_search_categorical.best_score_ > bayes_search_numeric.best_score_:
             best_params = bayes_search_categorical.best_params_
             best_score  = bayes_search_categorical.best_score_
        else:
            best_params = bayes_search_numeric.best_params_
            best_score  = bayes_search_numeric.best_score_

            print("Final Best Parameters:", best_params)
        # Best Parameters & Score
        print("Best Parameters:", best_params)
        print("Best Validation Accuracy:", best_score)
        return best_params
              