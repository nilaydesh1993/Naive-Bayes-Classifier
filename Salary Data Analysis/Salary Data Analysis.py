"""
Created on Wed May 13 12:18:18 2020
@author: DESHMUKH
NAIVE BAYES
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
pd.set_option('display.max_columns',None)

# ===============================================================================================
# Business Problem - Prepare a classification model using Naive Bayes for salary data.
# ===============================================================================================

salary_train = pd.read_csv("SalaryData_Train.csv")
salary_test = pd.read_csv("SalaryData_Test.csv")
salary_train.info()
salary_test.info()
salary_test.head()

# Value counts
salary_train.groupby('workclass').size()
salary_train.groupby('education').size()
salary_train.groupby('maritalstatus').size()
salary_train.groupby('occupation').size()
salary_train.groupby('relationship').size()
salary_train.groupby('race').size()
salary_train.groupby('sex').size()
salary_train.groupby('native').size()

############################ - Converting input variable into LabelEncoder - ############################

# All input categorical columns
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

# Counverting into Numerical Data by means of custome function and Lable Encoder
for i in string_columns:
    number = LabelEncoder()
    salary_train[i] = number.fit_transform(salary_train[i])
    salary_test[i] = number.fit_transform(salary_test[i])

##################################### - Splitting data in X and y - #####################################

X_train = salary_train.iloc[:,0:13]
y_train = salary_train.iloc[:,13]
X_test  = salary_test.iloc[:,0:13]
y_test  = salary_test.iloc[:,13]

########################################## - Fitting Model - ###########################################

# Model 1
# Multinomial Naive Bayes
smnb = MultinomialNB()
smnb.fit(X_train,y_train)

## Multinomial Model Accuracy
smnb.score(X_train,y_train) # 0.77
smnb.score(X_test,y_test)  # 0.77

# Model 2
# Gaussian Naive Bayes
sgnb = GaussianNB()
sgnb.fit(X_train,y_train)

## Gaussian Model Accuracy
sgnb.score(X_train,y_train) # 0.80
sgnb.score(X_test,y_test)  # 0.80

# From Above we can Conclude that Gaussian Naive Bayes Model gives us best result. So we are using it for future Predication.

# Prediction on Train & Test Data
pred_train = sgnb.predict(X_train)
pred_test = sgnb.predict(X_test)

# Confusion matrix of Train and Test
## Train
confusion_matrix_train = pd.crosstab(y_train,pred_train,rownames=['Actual'],colnames= ['Train Predictions']) 
sns.heatmap(confusion_matrix_train, annot = True, cmap = 'Blues',fmt='g')

## Test
confusion_matrix_test = pd.crosstab(y_test,pred_test,rownames=['Actual'],colnames= ['Test Predictions']) 
sns.heatmap(confusion_matrix_test, annot = True, cmap = 'Reds',fmt='g')

# Classification Report of test
print(classification_report(y_test,pred_test))


                          # ---------------------------------------------------- #