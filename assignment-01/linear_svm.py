# %%
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,roc_auc_score
import seaborn as sns
#importing the custome data handler
from data_handler.data_handler import DataHandler
from grid_search import *
from bayes_search import *

# %%
class LinearSVMClassifier:
    def __init__(self,C=1,Kernel="linear"):
        self.data_handler = DataHandler()
        self.model=SVC(C=1,kernel=Kernel)
    #loads the training data from the data handler as features and targets from the cleaned training data
    def load_train_data(self):
        print("Splitting training data into features and target variables")
        self.train_features,self.train_targets = self.data_handler.split_data(self.data_handler.train_data)
    
    #loads the validation data from the data handler as features and targets from the cleaned validation data
    def load_validation_data(self):
        print("Splitting validation data into features and target variables")
        self.validation_features,self.validation_targets = self.data_handler.split_data(self.data_handler.validation_data)
    
    #loads the test data from the data handler as features and targets from the cleaned test data
    def load_test_data(self):
        print("Splitting testing data into features and target variables")
        self.test_features,self.test_targets = self.data_handler.split_data(self.data_handler.test_data)
    
    #trains the model on the training data
    def train(self):
        print("Training the model with training data set")
        self.model.fit(self.train_features,self.train_targets)
    
    #predict the target for given features and returns the predicted targets use to predict the test data
    def predict(self,features):
        print("Predicting target varaibles using the features provided")
        return self.model.predict(features)
    
    #evaluates the model on the validation data
    def evaluate(self,DataFlag="Validation"):
        print("Evaluating the model's performance for {} dataset".format(DataFlag))
        if DataFlag=="Validation":
            features = self.validation_features
            targets = self.validation_targets
        elif DataFlag=="Test":
            features = self.test_features
            targets = self.test_targets
        predictions = self.predict(features)
        print("Accuracy: ",accuracy_score(targets,predictions))
        print("Classification Report: \n",classification_report(targets,predictions))
        print("Confusion Matrix: \n",confusion_matrix(targets,predictions))
        print("Roc Score : ",roc_auc_score(targets,predictions))
        self.plot_confusion_matrix(targets,predictions,DataFlag)
    

    #plot confusion matrix
    def plot_confusion_matrix(self,targets,predictions,dataFlag):
        cm = confusion_matrix(targets, predictions) 
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Survived", "Died"], yticklabels=["Survived", "Died"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix for {} data".format(dataFlag))
        plt.show()
    

# %%
linear_svm=LinearSVMClassifier()
linear_svm.load_train_data()
linear_svm.load_validation_data()
linear_svm.load_test_data()

# %%
linear_svm.train()

# %%
linear_svm.evaluate(DataFlag="Validation")

# %%
linear_svm.evaluate(DataFlag="Test")

# %%
gs=GridSearch()
gs_result=gs.tune_linear_svm(linear_svm.validation_features,linear_svm.validation_targets)

# %%
bs=BayesianSearch(n_iter=100)
bs_result=bs.tune_linear_svm(linear_svm.validation_features,linear_svm.validation_targets)

print("Grid Search Result : {0}\nBayesian Search Result : {1}".format(gs_result,bs_result))

# %%
linear_svm_optimized=LinearSVMClassifier(C=10)
linear_svm_optimized.load_train_data()
linear_svm_optimized.load_validation_data()
linear_svm_optimized.load_test_data()

# %%
linear_svm_optimized.train()

# %%
linear_svm_optimized.evaluate(DataFlag="Validation")

# %%
linear_svm_optimized.evaluate(DataFlag="Test")

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Example confusion matrices for different stages
conf_matrices = {
    "Unoptimized Validation": np.array([[49, 9], [8, 23]]),
    "Unoptimized Test": np.array([[95, 14], [23, 46]]),
    "Optimized Validation": np.array([[49, 9], [8, 23]]),
    "Optimized Test": np.array([[95, 14], [23, 46]])
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (title, cm) in enumerate(conf_matrices.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], 
                yticklabels=["Actual 0", "Actual 1"], ax=axes[i])
    axes[i].set_title(title)
    axes[i].set_xlabel("Predicted Label")
    axes[i].set_ylabel("True Label")

plt.tight_layout()
plt.show()



