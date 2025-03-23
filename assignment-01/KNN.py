# %%
#implementation of the KNN algorithm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
#importing the custome data handler
from data_handler.data_handler import DataHandler
from grid_search import *
from bayes_search import *

# %% [markdown]
# The `KNN` class implements the K-Nearest Neighbors algorithm with methods for 
# data loading, training, prediction, and evaluation. 
# It uses a custom `DataHandler` for managing data and a Scikit-learn KNN model.
# Key methods include `train()`, `predict()`, `evaluate()`, and `plot_confusion_matrix()`.

# %%
#class for the KNN algorithm
class KNN():
    def __init__(self,K=3):
        self.K=K 
        self.model = KNeighborsClassifier(n_neighbors=self.K)
        self.data_handler = DataHandler()
    
    #loads the training data from the data handler as features and targets from the cleaned training data
    def load_train_data(self):
        self.train_features,self.train_targets = self.data_handler.split_data(self.data_handler.train_data)
    
    #loads the validation data from the data handler as features and targets from the cleaned validation data
    def load_validation_data(self):
        self.validation_features,self.validation_targets = self.data_handler.split_data(self.data_handler.validation_data)
    
    #loads the test data from the data handler as features and targets from the cleaned test data
    def load_test_data(self):
        self.test_features,self.test_targets = self.data_handler.split_data(self.data_handler.test_data)
    
    #trains the model on the training data
    def train(self):
        self.model.fit(self.train_features,self.train_targets)
    
    #predict the target for given features and returns the predicted targets use to predict the test data
    def predict(self,features):
        return self.model.predict(features)
    
    #evaluates the model on the validation data
    def evaluate(self,DataFlag="Validation"):
        if DataFlag=="Validation":
            features = self.validation_features
            targets = self.validation_targets
        elif DataFlag=="Test":
            features = self.test_features
            targets = self.test_targets
        predictions = self.predict(features)
        print("Accuracy: ",accuracy_score(targets,predictions))
        print("Classification Report: ",classification_report(targets,predictions))
        print("Confusion Matrix: ",confusion_matrix(targets,predictions))
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



# %% [markdown]
# ### Creating a basic model with default parameters

# %%
print("Creating a basic KNN model with default parameters")
knn=KNN(K=3)
#load training data
knn.load_train_data()
#train model
knn.train()
#load validation data
knn.load_validation_data()

# %%
#evaluate model on validation data
print("Evaluating model on the validation dataset")
knn.evaluate(DataFlag="Validation")

# %%
#load test data
knn.load_test_data()
#evaluate model on test data
print("Evaluating model on the test dataset")
knn.evaluate(DataFlag="Test")

# %% [markdown]
# ### Performing model tuning to find optimal K

# %%
gs=GridSearch()
bs=BayesianSearch()
print("Applying Grid search to tune the K hyperparameter of the model")
result_gs=gs.tune_knn(x_train=knn.validation_features,y_train=knn.validation_targets,plot=True)

# %%
print("Applying Bayesian search to tune the K hyperparameter of the model")
result_bs=bs.tune_knn(x_train=knn.validation_features,y_train=knn.validation_targets)

# %%
print("Grid Search Results: ",result_gs)
print("Bayesian Search Results: ",result_bs)

# %% [markdown]
# ### New model with optimal K and checking accuracy on validation and test data

# %%
print("Creating a new KNN model with the tunned K parameter for max accuracy")
new_knn=KNN(K=12)
new_knn.load_train_data()
new_knn.train()


# %%
new_knn.load_validation_data()
print("Evaluating optimized model on the validation dataset")
new_knn.evaluate(DataFlag="Validation")

# %%
new_knn.load_test_data()
print("Evaluating optimized model on the test dataset")
new_knn.evaluate(DataFlag="Test")


