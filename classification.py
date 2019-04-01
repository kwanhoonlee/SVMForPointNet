from itertools import cycle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from pandas_ml import ConfusionMatrix


data = pd.read_csv('./svm_data/190327_SVM/SVM_Revised_KBIMS1.csv', header=0, index_col=None)
# X = data.drop(['types', 'bat_ids', 'global_ids', 'building'], 1)
X = data.drop(['types', 'bat_ids','type 2' ], 1)

Y = data['type 2']
le = LabelEncoder()
le.fit(Y)
encoded_Y = le.transform(Y)

Y = encoded_Y

# ### 2-1. Naive Bayes Classifier

clf = GaussianNB()
scores = cross_val_score(clf, X, Y, cv=10)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

# 2-2. Logistic Regression
clf = LogisticRegression(multi_class='ovr')
scores = cross_val_score(clf, X, Y, cv=10)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))


### 2-3. SVM
Y = [(0 if i=='IfcWallStandardCase' else (1 if i=='IfcBeam' else(2 if i=='IfcCovering' else (3 if i=='IfcColumn' else(4 if i=='IfcSlab' else(5 if i=='IfcDoor' else(6 if i=='IfcWindow' else 7))))))) for i in Y]
Y = label_binarize(Y, classes=[0, 1, 2, 3, 4, 5, 6, 7])
n_classes = Y.shape[1]

random_state = np.random.RandomState(0)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

clf = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True, random_state=random_state, C=1, gamma=0.1))
clf.fit(X_train, y_train)
y_score = clf.fit(X_train, y_train).decision_function(X_test)

print ("Accuracy: %0.4f" % clf.score(X_test, y_test)) 

## Run models WITH relational information¶

data = pd.read_csv('./building_rel_drop_duplicates.csv')
X = data.drop(['types', 'bat_ids', 'global_ids', 'building'], 1)
Y = data['types']

### 3-1. Naive Bayes Classifier

clf = GaussianNB()
scores = cross_val_score(clf, X, Y, cv=10)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))


### 3-2. Logistic Regression

clf = LogisticRegression(multi_class='ovr')
scores = cross_val_score(clf, X, Y, cv=10)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

### 3-3. SVM

random_state = np.random.RandomState(0)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=random_state)

clf = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True, random_state=random_state, C=10, gamma=0.1))
clf.fit(X_train, y_train)

print ("Accuracy: %0.4f" % clf.score(X_test, y_test)) 

### 3-3-1. Precision, Recall, F1-score

y_pred = clf.predict(X_test)

print (metrics.classification_report(y_test, y_pred))

y_pred = le.inverse_transform(y_pred)
y_test = le.inverse_transform(y_test)
### 3-3-2. Confusion matrix
# cm = ConfusionMatrix(np.array(y_test), y_pred)
cm = ConfusionMatrix(y_test, y_pred)

cm


### 3-3-3. Precison-recall curve

Y = [(0 if i=='IfcWallStandardCase' else (1 if i=='IfcBeam' else(2 if i=='IfcCovering' else (3 if i=='IfcColumn' else(4 if i=='IfcSlab' else(5 if i=='IfcDoor' else(6 if i=='IfcWindow' else 7))))))) for i in Y]
Y = label_binarize(Y, classes=[0, 1, 2, 3, 4, 5, 6, 7])
n_classes = Y.shape[1]
random_state = np.random.RandomState(0)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

clf = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True, random_state=random_state, C=10, gamma=0.1))
clf.fit(X_train, y_train)
y_score = clf.fit(X_train, y_train).decision_function(X_test)
print ("Accuracy: %0.4f" % clf.score(X_test, y_test)) 


precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []

l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

linestyles = cycle(['-', '--', '-.', ':'])
colors = ['aqua', 'darkorange', 'cornflowerblue', 'blue', 'purple', 'red', 'grey', 'yellow']
classes = ['IfcColumn', 'IfcBeam', 'IfcSlab', 'IfcWallStandardCase', 'IfcCovering', 'IfcDoor', 'IfcWindow', 'IfcRailing']

for i, line in zip(range(n_classes), linestyles):
    l, = plt.plot(recall[i], precision[i], color=colors[i], linestyle=line, lw=2)
    lines.append(l)
    labels.append('{0} (area = {1:0.2f})'
                  ''.format(classes[i], average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve (Rel O) ')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
plt.show()


### 3-3-4. ROU curve

# Compute ROC curve and ROC area for each class
n_classes = 8
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

classes = ['IfcColumn', 'IfcBeam', 'IfcSlab', 'IfcWallStandardCase', 'IfcCovering', 'IfcDoor', 'IfcWindow', 'IfcRailing']
linestyles = cycle(['-', '--', '-.', ':'])
colors = ['aqua', 'darkorange', 'cornflowerblue', 'blue', 'purple', 'red', 'grey', 'yellow']
for i, line in zip(range(n_classes), linestyles):
    plt.plot(fpr[i], tpr[i], color=colors[i], linestyle=line,
             label=' {0} (area = {1:0.2f})'
                   ''.format(classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class SVM result (rel O)')
plt.legend(loc="lower right")
plt.grid()
plt.show()


