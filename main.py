import pandas as pd
import matplotlib as plt
import tensorflow as tf
import sklearn as sk
from sklearn import svm
import tensorflow_datasets as tfds

class Titanic():

    def __init__(self):
        self.read_data()

    def read_data(self):
        #Bedrock function to enable 
        train_path = 'data/train.csv'
        test_path = 'data/test.csv'
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)
        print(self.train.head(1))

    def predict(self):
        x_train = self.train.loc[:,'fare']
        print(x_train.head(1))
        y_train = self.test.loc[:,'survived']
        clf = svm.SVC()
        clf.fit(x_train,y_train)


T = Titanic()
