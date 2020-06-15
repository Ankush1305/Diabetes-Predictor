#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# In[2]:


data=pd.read_csv('kaggle_diabetes.csv')


# In[3]:


data.head()


# In[4]:


data.dtypes


# In[5]:


data.shape


# In[7]:


data.isnull().sum()


# In[14]:


def histogram(data):
    for feature in data.columns:
        if feature!='Outcome':
            plt.hist(data[feature],bins =25)
            plt.title(feature)
            plt.show()


# In[15]:


histogram(data)


# In[16]:


data['Pregnancies'].unique()


# In[18]:


def boxplot(data):
    for feature in data.columns:
        if feature!='Outcome':
            plt.boxplot(data[feature])
            plt.title(feature)
            plt.show()


# In[19]:


boxplot(data)


# In[21]:


df_copy=data.copy()


# In[22]:


#replacing 0 values with nan
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)


# In[23]:


# Replacing NaN value by mean, median depending upon distribution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)


# In[24]:



X=df_copy.drop('Outcome',axis=1)
y=df_copy['Outcome']
X.head()



# In[25]:


y.head()


# In[26]:


#model selection
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=5)


# In[28]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix


# In[29]:


log_reg=LogisticRegression()


# In[31]:


log_reg.fit(X_train,y_train)


# In[32]:


pred=log_reg.predict(X_test)


# In[33]:


confusion_matrix(y_test,pred)


# In[34]:


accuracy_score(y_test,pred)


# In[35]:


# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)


# In[36]:


pred1=classifier.predict(X_test)


# In[37]:


accuracy_score(y_test,pred1)


# In[39]:


# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))


# In[ ]:




