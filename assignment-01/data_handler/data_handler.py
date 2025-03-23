import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE_NAME_TRAIN_DATA = os.path.join(BASE_DIR, "train_data.csv")
INPUT_FILE_NAME_TEST_DATA = os.path.join(BASE_DIR, "test_data.csv")
INPUT_FILE_NAME_VALIDATION_DATA = os.path.join(BASE_DIR, "val_data.csv")

class DataHandler:
    def __init__(self):
        self.train_data = pd.read_csv(INPUT_FILE_NAME_TRAIN_DATA)
        self.test_data = pd.read_csv(INPUT_FILE_NAME_TEST_DATA)
        self.validation_data = pd.read_csv(INPUT_FILE_NAME_VALIDATION_DATA)
        #dropping the passenger id column from each as we will not use it for training
        self.train_data=self.train_data.drop(columns=["PassengerId"])
        self.test_data=self.test_data.drop(columns=["PassengerId"])
        self.validation_data=self.validation_data.drop(columns=["PassengerId"])
    
    #function to split the data into features and target X is features and Y is target
    def split_data(self,data):
        X=data.drop(columns=["Survived"])
        Y=data["Survived"]
        return X,Y