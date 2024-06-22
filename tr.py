# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 13:06:19 2024

@author: Nazib
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from sklearn.utils import class_weight
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
X_train = train_df.drop(columns = ['Target'])
y_train = train_df['Target']

"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Separate features and target variable for test dataset
X_test = test_df
#y_test = test_df['Target']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform it
X_train_normalized = scaler.fit_transform(X_train)
# Transform the valid data using the fitted scaler
X_valid_normalized = scaler.transform(X_valid)
# Transform the test data using the fitted scaler
X_test_normalized = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=50, random_state=123)

# Train the classifier using extracted features and corresponding labels
rf_classifier.fit(X_train_normalized, y_train)

# Predict labels for validation data
y_pred_valid = rf_classifier.predict(X_valid_normalized)

# Calculate accuracy on the validation set
accuracy_valid = accuracy_score(y_valid, y_pred_valid)
print("Validation Accuracy:", accuracy_valid)

# Predict labels for test data
y_pred_test = rf_classifier.predict(X_test_normalized)

# Calculate accuracy on the test set
#accuracy_test = accuracy_score(y_valid, y_pred_test)
#print("Test Accuracy:", accuracy_test)
y_pred_test
"""
equilibre = train_df['Target'].value_counts()
print(equilibre)
plt.figure(figsize=(20,10))
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.pie(equilibre, labels=['Graduate','Dropout','Enrolled'], colors=['Blue','Green','Yellow'],autopct='%1.1f%%', textprops={'color': 'black'})
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

df_1 = train_df[train_df['Target'] == 'Enrolled']
df_2 = train_df[train_df['Target'] == 'Dropout']
from sklearn.utils import resample

df_1_upsample = resample(df_1, n_samples = 35000, replace = True, random_state = 123)
df_2_upsample = resample(df_2, n_samples = 35000, replace = True, random_state = 123)
df_0 = train_df[train_df['Target']=='Graduate'].sample(n = 35000, random_state=123)
train_df = pd.concat([df_0, df_1_upsample, df_2_upsample])


plt.figure(figsize= (10,10))
my_circle = plt.Circle((0,0), 0.7, color = 'white') 
plt.pie(train_df['Target'].value_counts(), labels=['Graduate','Dropout','Enrolled'], colors=['Blue','Green','Yellow'])
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

X_train = train_df.drop(columns = ['Target'])
y_train = train_df['Target']

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Separate features and target variable for test dataset
X_test = test_df
#y_test = test_df['Target']

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform it
X_train_normalized = scaler.fit_transform(X_train)
# Transform the valid data using the fitted scaler
X_valid_normalized = scaler.transform(X_valid)
# Transform the test data using the fitted scaler
X_test_normalized = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=123)

# Train the classifier using extracted features and corresponding labels
rf_classifier.fit(X_train_normalized, y_train)

# Predict labels for validation data
y_pred_valid = rf_classifier.predict(X_valid_normalized)

# Calculate accuracy on the validation set
accuracy_valid = accuracy_score(y_valid, y_pred_valid)
print("Validation Accuracy:", accuracy_valid)

# Predict labels for test data
y_pred_test = rf_classifier.predict(X_test_normalized)

# Calculate accuracy on the test set
#accuracy_test = accuracy_score(y_valid, y_pred_test)
#print("Test Accuracy:", accuracy_test)
print(y_pred_test)