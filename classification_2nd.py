import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


data = pd.read_csv('./6ifc_2nd_v3.csv')
data2 = data.drop_duplicates(subset=['X', 'Y', 'Z', 'areas', 'ax1s', 'ax2s', 'gyrations', 'volumes', 'subtype'], keep='first')

windowA = data[data['subtype']=='IfcWindow-A']
windowB = data[data['subtype']=='IfcWindow-B']
window = pd.concat([windowA, windowB])
window = window.drop_duplicates(subset=['X', 'Y', 'Z', 'areas', 'ax1s', 'ax2s', 'gyrations', 'volumes', 'subtype'], keep='first')

doorA = data[data['subtype']=='IfcDoor-A']
doorB = data[data['subtype']=='IfcDoor-B']
door = pd.concat([doorA, doorB])
door = door.drop_duplicates(subset=['X', 'Y', 'Z', 'areas', 'ax1s', 'ax2s', 'gyrations', 'volumes', 'subtype'], keep='first')

columnA = data[data['subtype']=='IfcColumn-A']
columnB = data[data['subtype']=='IfcColumn-B']
column = pd.concat([columnA, columnB])
column = column.drop_duplicates(subset=['X', 'Y', 'Z', 'areas', 'ax1s', 'ax2s', 'gyrations', 'volumes', 'subtype'], keep='first')

## Run models WITHOUT relational information
def classification_2nd_without_rel(data):

    def get_upsampled(X, Y):
        ros = RandomOverSampler()
        X_upsampled, Y_upsampled = ros.fit_sample(X, Y)
        return X_upsampled, Y_upsampled

    X = data.drop(['subtype', 'bat_ids', 'global_ids', 'building', 'IfcDoor', 'IfcWallStandardCase', 'IfcWindow'], 1)
    Y = data['subtype']

    newX, newY = get_upsampled(X, Y)

    X_train, X_test, y_train, y_test = train_test_split(newX, newY, test_size=0.2, random_state=0)

    clf = svm.SVC(kernel='rbf', C=10, gamma=0.1, probability=True)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    return acc


door_2nd_result = classification_2nd_without_rel(door)
window_2nd_result = classification_2nd_without_rel(window)
column_2nd_result = classification_2nd_without_rel(column)
print ("door_2nd classification result:", door_2nd_result)
print ("window_2nd classification result:", window_2nd_result)
print ("column_2nd classification result:", column_2nd_result)


## Run models WITH relational information
def classification_2nd_with_rel(data):

    def get_upsampled(X, Y):
        ros = RandomOverSampler()
        X_upsampled, Y_upsampled = ros.fit_sample(X, Y)
        return X_upsampled, Y_upsampled

    X = data.drop(['subtype', 'bat_ids', 'global_ids', 'building'], 1)
    Y = data['subtype']

    newX, newY = get_upsampled(X, Y)

    X_train, X_test, y_train, y_test = train_test_split(newX, newY, test_size=0.2, random_state=0)

    clf = svm.SVC(kernel='rbf', C=10, gamma=0.1, probability=True)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)

    return acc

door_2nd_result = classification_2nd_with_rel(door)
window_2nd_result = classification_2nd_with_rel(window)
column_2nd_result = classification_2nd_with_rel(column)
print ("door_2nd classification result(rel):", door_2nd_result)
print ("window_2nd classification result(rel):", window_2nd_result)
print ("column_2nd classification result(rel):", column_2nd_result)