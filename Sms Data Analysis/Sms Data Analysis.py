"""
Created on Wed May 13 12:18:18 2020
@author: DESHMUKH
NAIVE BAYES
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

# ===============================================================================================
# Business Problem - Build a naive Bayes model on the data set for classifying the ham and spam.
# ===============================================================================================

sms = pd.read_csv("sms_raw_NB.csv",encoding = "ISO-8859-1")
sms.info()
sms.head()
sms.tail()
sms.isnull().sum()

# Count of Ham and Spam
sms.groupby('type').describe()

# Checking Percentage of Output classes with the help Value Count.
(sms['type'].value_counts())/len(sms)*100 # Spam = 87% , Ham = 13% 

####################################### - Data Preprocessing - #######################################

# Opening Custom Build Stopword Dataset
with open("stopwords_en.txt","r") as sw: 
    stopwords = sw.read()    
    
# Coverting them individual Words. 
stopwords = stopwords.split("\n")

# Custome Build function to remove Stopword,Special symbol and all other unneccesary things
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if not word in stopwords:
            w.append(word)
    return (" ".join(w))

# Passing Data set input variable from above function
sms.text = sms.text.apply(cleaning_text)

################################ - Splitting data in train and test - #################################

# Stratified Sampling
X_train,X_test,y_train,y_test = train_test_split(sms.text, sms.type, test_size=0.3, random_state = False, stratify = sms.type)

# Rechecking Percentage of Output classes by using values counts. 
(y_train.value_counts()/len(y_train))*100 # Spam = 87%, Ham = 13% Percentage of sample Output classes simillier to population. 

###################################### - Count Vectorization - #######################################

# Input data is in document (text) format so we have to use Count Vectorization to convert it into Matrix.
# Count Vectorization
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_test_count = v.transform(X_test)

# Convertion into Array for Gaussian Naive Bayes Model.
X_train_count_array = X_train_count.toarray()
X_test_count_array = X_test_count.toarray()

######################################### - Fitting Model- ###########################################

# Model 1
# Multinomial Naive Bayes
smnb = MultinomialNB()
smnb.fit(X_train_count,y_train)

## Multinomial Model Accuracy
smnb.score(X_train_count,y_train) # 0.99
smnb.score(X_test_count,y_test)  # 0.98

# Model 2
# Gaussian Naive Bayes
sgnb = GaussianNB()
sgnb.fit(X_train_count_array,y_train)

## Gaussian Model Accuracy
sgnb.score(X_train_count_array,y_train) # 0.90
sgnb.score(X_test_count_array,y_test)  # 0.85

# From Above we can Conclude that Multinomial Naive Bayes Model gives us best result. So we are using it for future Predication.

# Prediction on Train & Test Data
pred_train = smnb.predict(X_train_count)
pred_test = smnb.predict(X_test_count)

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
          
