# ProjectName : SVMForPointNet
# FileName : SVM_Class
# Created on : 27/03/20193:52 PM
# Created by : KwanHoon Lee
import itertools
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import svm

class bim_svm():

    def __init__(self, config):
        self.config = config


    def preprocessing_rawdata(self):

        raw = pd.read_csv(self.config['raw'], header=0, index_col=None)

        X = raw[self.config['independents']]
        Y = raw[self.config['dependent']]

        print(raw.describe())
        return X, Y


    def separating_dataset(self, X, Y):

        random_state = np.random.RandomState(self.config['random_state'])

        # X, Y = self.preprocessing_rawdata()

        trainX, testX, trainY, testY = train_test_split(X, Y, test_size=self.config['test_size'])


        return trainX, testX, trainY, testY


    def labeling(self, Y):
        le = LabelEncoder()
        le.fit(Y)
        encoded_Y = le.transform(Y)

        return le, encoded_Y


    def learning(self, x, y):

        random_state = np.random.RandomState(self.config['random_state'])

        clf = OneVsRestClassifier(svm.SVC(kernel=self.config['kernel'], probability=True, random_state=random_state, C=self.config['C'], gamma=self.config['gamma']))
        clf.fit(x, y)

        return clf


    def score(self, clf, testX, testY):
        accuracy = clf.score(testX, testY)

        print("Accuracy: %0.4f" % clf.score(testX, testY))

        return accuracy


    def prediction(self, clf, Y):
        predY = clf.predict(Y)

        return predY

    def inverse_labeling(self, le, Y):
        inversed_y = le.inverse_transform(Y)

        return inversed_y


    def confusionMatrix(self, Y, predY):
        cm = confusion_matrix(Y, predY)

        return cm



    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion Matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        nowtxt = datetime.now().strftime('%m-%d_%H_%M')

        plt.savefig(self.config['plt_path'] + title + "_"+ nowtxt + ".png", dpi=600)
        plt.clf()
        plt.close()
