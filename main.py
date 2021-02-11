import pandas as pd
import numpy as np
import matplotlib as plt
import tensorflow as tf
import sklearn as sk
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import tensorflow_datasets as tfds

class Titanic():

    def __init__(self):
        #Constructor
        self.read_data()

    def read_data(self):
        #Bedrock function to enable 
        train_path = 'data/train.csv'
        test_path = 'data/test.csv'
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)

    def analyze_numeric(self):
        #Looks for most important feeatures
        raw_train = self.train.iloc[:,self.train.columns != 'Survived'].\
            select_dtypes(include=[np.float64,np.int64, np.float32,np.int32]).astype(np.float64)
        raw_val = self.train.loc[:,'Survived'].values

        #Params
        param_grid = \
        {
        'pca__n_components': [1,2,4,5],
        }

        #Split my labeled data, hurts but what can you do.
        x_train, x_test, y_train, y_test = train_test_split(raw_train, raw_val ,test_size=0.3, random_state=42)
        
        #Load the PCA
        pca = PCA()

        



    def predict(self):
        #Simple SVM implementation
        print('Started Prediction:')
        raw_train = self.train.loc[:,'Fare'].values.reshape(-1,1)
        raw_val = self.train.loc[:,'Survived'].values.reshape(-1,1)

        #Split my labeled data, hurts but what can you do.
        x_train, x_test, y_train, y_test = train_test_split(raw_train, raw_val ,test_size=0.3, random_state=42)
        clf = svm.SVC().fit(x_train,y_train)
        print(clf.score(x_test, y_test))
        '''
        y_test = clf.predict(x_test)
        scores = cross_val_score(clf,x_test, y_test, cv=5)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        '''
        print('\n\n')


T = Titanic()
T.analyze_numeric()
