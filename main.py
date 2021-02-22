import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn as sk
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
        self.train = pd.read_csv(train_path).fillna(value = 0)
        self.test = pd.read_csv(test_path).fillna(value = 0)

    def buildPipeline(self):
        return Pipeline(steps=[('standardscaler', StandardScaler())])

    def preprocess(self):
        return 0

    def convertSex(self,sex):
        if sex == "male":
            return 0
        else:
            return 1

    def analyze_numeric(self):
        #Looks for most important feeatures

        self.train['Sex'] = self.train.iloc['sex'].apply(self.convertSex())
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
        
        self.plotKMeans(raw_train)
        
        model = kmeans.fit(x_train)
        model.predict(x_test)




    def plotKMeans(self,raw_train):
        '''
        Takes 
        '''
        reduced_data = StandardScaler().fit(raw_train).transform(raw_train[:,['Pclass','Sex']])
        #reduced_data = PCA(n_components=2).fit_transform(raw_train)
        kmeans = KMeans(init="k-means++", n_clusters=3, n_init=4)
        kmeans.fit(reduced_data)

        h = 0.2
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation="nearest",
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired, aspect="auto", origin="lower")

        plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
                    color="w", zorder=10)
        plt.title("K-means clustering on the digits dataset (PCA-reduced data)\n"
                  "Centroids are marked with white cross")
        plt.xlabel('Sex')
        plt.ylabel('pclass')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()

        return 0


    def performKMeans(self):
        return 0



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
