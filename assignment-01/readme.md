# Assignment 01 Binary classification of passenger survival using the given data set

## Understanding the problem
Given the dataset '.data/data.csv' implement the below algorithms to solve a binary classification problem where the aim is to predict "given the specific features" whether the person survives or not.

## TODO

### Algorithms to implement

1. Naive Bayes
2. Logistic Regression
3. K-Nearest Neigbor (KNN)
4. Support Vector Machines (SVM)

Implement all the above mentioned algorithms along with hyperparameter tuning use the below methods

1. Grid Search
2. Bayesian Search

In case of SVM, hyper parameter search must include "C", "Gamma" and the "Kernels" (linear and RBF)

### Load and process the dataset

- [ ] Use CSV as file separators while opening and loading the dataset
- [ ] Handle the missing values
- [ ] Encode categorical features (One-Hot encoding)
- [ ] Standardize numerical features
- [ ] Split data into **70% training** **10% validation** and **20% test** sets

### Test set

- Train split  should be used for model training
- Validation split should be used to compare models for different hyperparameters
- Based on the best hyperparameters obtained, use test split to report the model's final performance
