#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %% [code]
##########################################################
# 1. IMPORT ALL PACKAGES
##########################################################
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import math
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt #for plotting
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import multilabel_confusion_matrix
from math import sqrt
# importing mean() 
from statistics import mean
from IPython import get_ipython

# %% [code]
##########################################################
# 2. LOAD DATASET
##########################################################
data = pd.read_csv("letter-recognition.csv",header=0)# header 0 means the first row is name of the coloumn 

 
# View sample data
data.head(10) 

# %% [code]
# Plot distribution
sns.countplot(data['letter'])

# %% [code]
# 3. SHARE TO TEST AND TRAIN DATA
##########################################################
x = data.iloc[:, 1:]
y = data['letter'].tolist()
print(x)

# Select 4000 rows data as a testing dataset
x_test = x.iloc[0:4000, :].values.astype('float32') # all pixel values 
y_test = y[0:4000] # Select label for testing data
x_train = x.iloc[4000:, :].values.astype('float32') # all pixel values 
y_train = y[4000:]

# # Share test and train data
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# %% [code]
##########################################################
# 4. PREVIEW 10 IMAGE DATA
##########################################################
#get_ipython().run_line_magic('matplotlib', 'inline')
# 10 selected rows
plt.figure(figsize=(12,10))
x_cor, y_cor = 4, 4
for i in range(10):  
    arr_img = np.asarray(x_train[i].reshape((4,4)));
    plt.subplot(y_cor, x_cor, i+1)
    #plt.imshow(arr_img, interpolation='nearest')
    plt.imshow(arr_img, cmap='gray') # Displaying a grayscale image
plt.show()

# %% [code]
##########################################################
# 5. FEATURE NORMALIZATION FOR BOTH (TEST & TRAIN)
##########################################################
# Proceed to normalize the features because the pixel intensities are currently between the range of 0 and 255
print((min(x_train[2]), max(x_train[2])))

# Normalizing the data
x_train = x_train/255.0
x_test = x_test/255.0

# Printing the shape of the Datasets
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# %% [code]
##########################################################
# 6. TRAIN RANDOM FOREST ALGORITHM
##########################################################
# Create a RandomForestClassifier object with the parameters over the data
# n_estimators (default=100) = the number of trees in the forest.
# max_depth (default=None) = the maximum depth of the tree.
model_clf = RandomForestClassifier(n_estimators=500, max_depth=2, random_state=0)

# Train the Random Forest algorithm
model_clf.fit(x_train, y_train)

# %% [code]
##########################################################
# 7. APPLY THE TRAINED LEARNER TO TEST NEW DATA
##########################################################
# Apply the trained perceptron to make prediction of test data
y_pred = model_clf.predict(x_test)

# %% [code]
##########################################################
# 8. MULTI-CLASS CONFUSION MATRIX FOR EACH CLASS
##########################################################

# Actual and predicted classes
lst_actual_class = y_test
lst_predicted_class = y_pred

# Class = Label A-Z
lst_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' ]


# Compute multi-class confusion matrix
arr_out_matrix = multilabel_confusion_matrix(lst_actual_class, lst_predicted_class, labels=lst_classes)
# Temp store results
store_sens = [];
store_spec = [];
store_acc = [];
store_bal_acc = [];
store_prec = [];
store_fscore = [];
store_mcc = [];
for no_class in range(len(lst_classes)):
    arr_data = arr_out_matrix[no_class];
    print("Print Class: {0}".format(no_class));

    tp = arr_data[1][1]
    fp = arr_data[0][1]
    tn = arr_data[0][0]
    fn = arr_data[1][0]
    
    
    sensitivity = round(tp/(tp+fn), 3);
    specificity = round(tn/(tn+fp), 3);
    accuracy = round((tp+tn)/(tp+fp+tn+fn), 3);
    balanced_accuracy = round((sensitivity+specificity)/2, 3);
    
    precision = round(tp/(tp+fp), 3);
    f1Score = round((2*tp/(2*tp + fp + fn)), 3);
    
    x = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
    MCC = round(((tp * tn) - (fp * fn)) / sqrt(x), 3)
    store_sens.append(sensitivity);
    store_spec.append(specificity);
    store_acc.append(accuracy);
    store_bal_acc.append(balanced_accuracy);
    store_prec.append(precision);
    store_fscore.append(f1Score);
    store_mcc.append(MCC);
    print("TP={0}, FP={1}, TN={2}, FN={3}".format(tp, fp, tn, fn));
    print("Sensitivity: {0}".format(sensitivity));
    print("Specificity: {0}".format(specificity));
    print("Accuracy: {0}".format(accuracy));
    print("Balanced Accuracy: {0}".format(balanced_accuracy));
    
    print("Precision: {0}".format(precision));
    print("F1-Score: {0}".format(f1Score));
    print("MCC: {0}\n".format(MCC));


# %% [code]
##########################################################
# 9. OVERALL - FINAL PREDICTION PERFORMANCE
##########################################################

print("Overall Performance Prediction:");
print("Sensitivity: {0}%".format(round(mean(store_sens)*100, 4)));
print("Specificity: {0}%".format(round(mean(store_spec)*100, 4)));
print("Accuracy: {0}%".format(round(mean(store_acc)*100, 4)));
print("Balanced Accuracy: {0}%".format(round(mean(store_bal_acc)*100, 4)));
print("Precision: {0}%".format(round(mean(store_prec)*100, 4)));
print("F1-Score: {0}%".format(round(mean(store_fscore)*100, 4)))
print("MCC: {0}\n".format(round(mean(store_mcc), 4)))

