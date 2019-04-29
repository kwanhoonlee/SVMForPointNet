# ProjectName : SVMForPointNet
# FileName : SVM_Class
# Created on : 27/03/20193:52 PM
# Created by : KwanHoon Lee
import itertools
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, roc_curve, auc
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


    def getPrecision_Recall(self, testY, scoreY):
        precision = dict()
        recall = dict()
        avg_precision = dict()

        for i in range(testY.shape[1]):
            precision[i], recall[i], _ = precision_recall_curve(testY[:, i], scoreY[:, i])
            avg_precision[i] = average_precision_score(testY[:, i], scoreY[:, i])

        return precision, recall, avg_precision

    def getMicroPrecision_Recall(self, testY, scoreY):
        p, r, ap = self.getPrecision_Recall(testY, scoreY)
        p["micro"], r["micro"], _ = precision_recall_curve(testY.ravel(), scoreY.ravel())
        ap["micro"] = average_precision_score(testY, scoreY, average="micro")

        print('Average precision score, micro-averaged over all classes: {0:0.2f}'
              .format(ap["micro"]))

        return p, r, ap

    def transform_using_onehotencoder(self, Y):
        encoder = OneHotEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)

        return encoder, encoded_Y.toarray()

    def plot_precision_recall_curve(self, testY, scoreY, classes):
        plt.clf()
        p, r, ap = self.getMicroPrecision_Recall(testY, scoreY)
        plt.figure(figsize=(7,8))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []

        l, = plt.plot(r['micro'], p['micro'], color='gold', lw=2)
        lines.append(l)
        labels.append('Micro-Average Precision-Recall (area = {0:0.2f})'
                      ''.format(ap['micro']))

        linestyles = itertools.cycle(['-', '--', '-.', ':'])
        colors = ['tomato', 'darkorange', 'green', 'blue',] #'purple','red', 'grey', 'yellow'
        for i, line in zip(range(len(classes)), linestyles):
            l, = plt.plot(r[i], p[i], color=colors[i], linestyle=line, lw=2)

            lines.append(l)
            labels.append('{0} (area = {1:0.2f})'
                          ''.format(classes[i], ap[i]))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Rel O) ')
        plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
        nowtxt = datetime.now().strftime('%m-%d_%H_%M')

        plt.savefig(self.config['plt_precision_recall_path'] +"Precision_Recall_Curve_"+ nowtxt + ".png", dpi=600 )
        # plt.show()

        return p, r, ap

    def plot_roc_curve(self, testY, scoreY, classes):
        plt.clf()
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(testY.shape[1]) :
            fpr[i], tpr[i], _ = roc_curve(testY[:, i ], scoreY[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])


        linestyles = itertools.cycle(['-', '--', '-.', ':'])

        colors = ['tomato', 'darkorange', 'green', 'blue',]
        for i, line in zip(range(testY.shape[1]), linestyles):
            plt.plot(fpr[i], tpr[i], color=colors[i], linestyle=line, label=' {0} (area = {1:0.2f})'
                     ''.format(classes[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-Class SVM Result (rel O)')
        plt.legend(loc="lower right")
        plt.grid()
        # plt.show()
        nowtxt = datetime.now().strftime('%m-%d_%H_%M')


        plt.savefig(self.config['plt_roc_curve_path'] +"ROC Cureve "+ nowtxt + ".png", dpi=600 )

        return fpr, tpr, roc_auc