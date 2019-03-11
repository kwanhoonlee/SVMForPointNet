import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

data = pd.read_csv('./building_drop_duplicates.csv')
X = data.drop(['types', 'bat_ids', 'global_ids', 'building'], 1)
Y = data['types']

tmp = data.groupby('types').mean()

data_prep_XY = data[data['X'] <20]
data_prep_XY = data_prep_XY[data_prep_XY['Y'] <20]
data_prep_ax2s = data[data['ax2s']<300]

x = data_prep_ax2s['ax1s']
y = data_prep_ax2s['ax2s']

cluster = pd.factorize(data_prep_ax2s['types'])[0]
cls =pd.factorize(data_prep_ax2s['types'])

fig, ax = plt.subplots()
labels=['IfcColumn', 'IfcBeam', 'IfcSlab', 'IfcWallStandardCase', 'IfcCovering', 'IfcDoor', 'IfcWindow', 'IfcRailing']

ax.scatter(x[cluster==0], y[cluster==0], s=100, marker='o', label='IfcColumn')
ax.scatter(x[cluster==1], y[cluster==1], s=100, marker='v', label='IfcBeam')
ax.scatter(x[cluster==2], y[cluster==2], s=100, marker='x', label='IfcSlab')
ax.scatter(x[cluster==3], y[cluster==3], s=100, marker='<', label='IfcWallStandardCase')
ax.scatter(x[cluster==4], y[cluster==4], s=100, marker='d', label='IfcCovering')
ax.scatter(x[cluster==5], y[cluster==5], s=100, marker='*', label='IfcDoor')
ax.scatter(x[cluster==6], y[cluster==6], s=100, marker='D', label='IfcWindow')
ax.scatter(x[cluster==7], y[cluster==7], s=100, marker='s', label='IfcRailing')
ax.legend()