import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from google.colab import drive
import argparse
import logging
import pathlib
import functools

import cv2
import torch
from torchvision import transforms

drive.mount('/content/drive')

data = '/content/drive/MyDrive/Senior Project/eczema_non_eczema_resnet152_makingsure_features_balanced_segmented.csv'

df = pd.read_csv(data)

# check distribution of target_class column

df['Class Label'].value_counts()

X = df.drop(['Class Label'], axis=1)
y = df['Class Label']

# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=0) #check splitting in past papersX_train

y_test.value_counts()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), pca.explained_variance_ratio_, marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Component')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), pca.explained_variance_ratio_.cumsum(), marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by Number of Principal Components')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], alpha=0.2)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Scatter Plot of First Two Principal Components')
plt.show()

#X_train=X_train.drop(['Image Path'],axis=1)
#X_test=X_test.drop(['Image Path'],axis=1)
# Convert the NumPy array to a Pandas DataFrame
X_train = pd.DataFrame(X_train)
cols = X_train.columns

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Import required libraries
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#svc = SVC(C=10, gamma=0.01)
svc=SVC(kernel='poly',C=1)
# Fit classifier to training set
svc.fit(X_train, y_train)

# Make predictions on test set
y_pred = svc.predict(X_test)

# Compute performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
# Print metrics
print('Model Performance Metrics with',svc.kernel,'kernel, C=',svc.C,)# and',pca.n_components,'features: ')
print('------------------------------------')
print('Accuracy : {0:0.4f}'.format(accuracy))
print('Precision: {0:0.4f}'.format(precision))
print('Recall   : {0:0.4f}'.format(recall))
print('F1-score : {0:0.4f}'.format(f1))
print('Confusion Matrix:')
# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=conf_matrix, columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

def highlight_predictions(row):
    color = 'green' if row['Actual'] == row['Predicted'] else 'red'
    return ['color: %s' % color] * len(row)

styled_results = results.style.apply(highlight_predictions, axis=1)


styled_results

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
pca = PCA(n_components=100, random_state=48)
svm = SVC(kernel='poly', C=1)

# Using pipeline to perform segmentation/feature extraction/dimensionality reduc/normalization/classif
pipeline = Pipeline([
    ('pca', pca),
    ('scaler', scaler),
    ('svm', svm)
])

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


pipeline.fit(X_train, y_train)

# Evaluating the classifier
print("Test Accuracy:", pipeline.score(X_test, y_test))

X_single = X_test.iloc[7].values  # Convert Series to numpy array
Y_single=y_test.iloc[7]
X_single_reshaped = X_single.reshape(1, -1)  # Reshape your data
prediction = pipeline.predict(X_single_reshaped)
print("Prediction for the single sample:", prediction)
print("Actual value: ",Y_single )

model=joblib.load('my_model_pipeline.pkl')

X_single_reshaped.shape

pred=model.predict(X_single_reshaped)
print(pred)

# import GridSearchCV
from sklearn.model_selection import GridSearchCV


# import SVC classifier
from sklearn.svm import SVC


# instantiate classifier with default hyperparameters with kernel=rbf, C=1.0 and gamma=auto
svc=SVC()



# declare parameters for hyperparameter tuning
parameters = [ {'C':[1, 10, 100, 1000], 'kernel':['linear']},
               {'C':[1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
               {'C':[1, 10, 100, 1000], 'kernel':['poly'], 'degree': [2,3,4] ,'gamma':[0.01,0.02,0.03,0.04,0.05]}
              ]




grid_search = GridSearchCV(estimator = svc,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           verbose=0)


grid_search.fit(X_train, y_train)

# examine the best model


# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))


# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))


# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))

# calculate GridSearch CV score on test set

print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))


