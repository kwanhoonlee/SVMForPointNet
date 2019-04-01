# ProjectName : SVMForPointNet
# FileName : 190327_svm
# Created on : 27/03/20194:02 PM
# Created by : KwanHoon Lee


config = {
    'name':'All',
    'raw':'./svm_data/190401_SVM/concat_all.csv',
    'independents':['X', 'Y', 'Z', 'areas', 'ax1s', 'ax2s', 'gyrations', 'volumes'],
    'dependent':'type 2',

    'random_state': 3,
    'test_size': 0.3,
    'kernel':'rbf',
    'C': 1,
    'gamma':0.1,
    'n_split': 5,


    'plt_path':'./result/plt/confusion_matrix/190401/'

}


import pandas as pd
from SVM_Class import bim_svm


bs = bim_svm(config)


X, Y = bs.preprocessing_rawdata()


le, Y = bs.labeling(Y)
trainX, testX, trainY, testY = bs.separating_dataset(X, Y)


model = bs.learning(trainX, trainY)
accuracy = bs.score(model, testX, testY)
predY = bs.prediction(model, testX)


y = bs.inverse_labeling(le, testY)
predY = bs.inverse_labeling(le, predY)


cm = bs.confusionMatrix(y, predY)


bs.plot_confusion_matrix(cm, le.classes_, False, config['name'] +' Confusion Matrix, Accuracy ' + str(round(accuracy, 4)))
bs.plot_confusion_matrix(cm, le.classes_, True, config['name'] +' Normalized Confusion Matrix, Accuracy ' + str(round(accuracy, 4)) )


#
# data1 = pd.read_csv('./svm_data/190327_SVM/SVM_Revised_YONSEI.csv', header=0)
# data2 = pd.read_csv('./svm_data/190327_SVM/SVM_Revised_KBIMS1.csv', header=0)
# data3 = pd.read_csv('./svm_data/190327_SVM/SVM_Revised_KBIMS2.csv', header=0)

# pd.concat([data1, data2, data3]).to_csv('./svm_data/concat_all.csv')
