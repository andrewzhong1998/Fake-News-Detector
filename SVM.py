#!/usr/bin/env python
# coding: utf-8

# <b>open file<b>

# In[1]:


import numpy as np
from sklearn.preprocessing import normalize


# In[2]:


train = np.load('/Users/yuji/Desktop/COMP562/FinalProject/data/training_features1.npy')
train = normalize(train, axis = 1)

train_label = np.load('/Users/yuji/Desktop/COMP562/FinalProject/data/training_labels1.npy')


test = np.load('/Users/yuji/Desktop/COMP562/FinalProject/data/test_features.npy')
test = normalize(test, axis = 1)

test_label = np.load('/Users/yuji/Desktop/COMP562/FinalProject/data/test_labels.npy')


# In[3]:


train_small = train[:1000, :]
train_label_small = train_label[:1000]
test_small = test[1000:2000, :]
test_label_small = test_label[1000:2000]
print(test.shape)


# In[4]:


from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear', C = 1000)  
svclassifier.fit(train_small, train_label_small)


# In[24]:


from sklearn.metrics import classification_report, confusion_matrix  
test_output_small = svclassifier.predict(test_small)
print(confusion_matrix(test_output_small, test_label_small))  
print(classification_report(test_output_small, test_label_small))  

