#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
import pickle
import os


# In[31]:


os.chdir(r'C:\Users\Mummy\Documents\data source')


# In[32]:


df=pd.read_csv('train.csv')


# In[36]:


X=df.drop(columns=['price_range'])   #train test split
y=df['price_range']


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[39]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'mobileprice.pkl'
pickle.dump(classifier, open(filename, 'wb'))


# In[ ]:




