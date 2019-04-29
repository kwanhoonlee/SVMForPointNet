# ProjectName : SVMForPointNet
# FileName : 190429_svm
# Created on : 29/04/20194:56 PM
# Created by : KwanHoon Lee



import pandas as pd
from SVM_Class import bim_svm

config = {
    'name':'190429',
    'raw':'./svm_data/190429_SVM/SVM_Train.csv',
    'independents':['X', 'Y', 'Z', 'areas', 'ax1s', 'ax2s', 'gyrations', 'volumes'],
    'dependent':'type 2',

    'random_state':3,
    'test_size':0.3,
    'kernel':'rbf',
    'C':1,
    'gamma':0.1,
    'n_split':5,

    'plt_path':'./result/plt/confusion_matrix/190429/',
    'plt_precision_recall_path':'./result/plt/precision_recall_curve/190429/',
    'plt_roc_curve_path':'./result/plt/roc_curve/190429/'
}

config_for_test = {
    'raw':'./svm_data/190429_SVM/SVM_Test.csv',
    'independents':['X', 'Y', 'Z', 'areas', 'ax1s', 'ax2s', 'gyrations', 'volumes'],
    'dependent':'type 2',
}

bs = bim_svm(config)
bs_test = bim_svm(config_for_test)

trainX, trainY = bs.preprocessing_rawdata()
testX, testY = bs_test.preprocessing_rawdata()

encoder, trainY = bs.transform_using_onehotencoder(trainY.values.reshape(-1,1))
_, testY = bs.transform_using_onehotencoder(testY.values.reshape(-1,1))
# le, trainY = bs.labeling(trainY)
# _, testY = bs.labeling(testY)


model = bs.learning(trainX, trainY)
scoreY = model.decision_function(testX)
accuracy = bs.score(model, testX, testY)
predY = bs.prediction(model, testX)


# y = bs.inverse_labeling(le, testY)
# predY = bs.inverse_labeling(le, predY)

y = bs.inverse_labeling(encoder, testY)
predY = bs.inverse_labeling(encoder, predY)

cm = bs.confusionMatrix(y, predY)


# classes = le.classes_
classes = encoder.categories_[0][:]

bs.plot_confusion_matrix(cm, classes, False, config['name'] +' Confusion Matrix, Accuracy ' + str(round(accuracy, 4)))
bs.plot_confusion_matrix(cm, classes, True, config['name'] +' Normalized Confusion Matrix, Accuracy ' + str(round(accuracy, 4)) )


p, r, ap = bs.getMicroPrecision_Recall(testY, scoreY)
bs.plot_precision_recall_curve(testY, scoreY, classes)
bs.plot_roc_curve(testY, scoreY, classes)

