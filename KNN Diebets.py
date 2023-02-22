#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# In[2]:



df=pd.read_csv(r"C:\Users\ANJANA\Desktop\ML\diabetes.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.describe()


# In[5]:


df.isna().sum()


# In[6]:


df.columns


# In[7]:


#to select input columns
x=df.iloc[::-1].values
x


# In[8]:


#to select output columns
y=df.iloc[:,-1].values
y


# In[9]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
x_train


# In[10]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
x_test


# In[11]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
y_train


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)
y_test


# In[13]:


#standard scalar\
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_train


# In[14]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_test)
x_test=scaler.transform(x_test)
x_test


# In[15]:


#model creation
classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y_train)


# In[16]:


y_pred=classifier.predict(x_test)
y_pred
#print(classifier.predict([[5,148,72,35,94,28,0.167,25,32]]))


# In[17]:


#performance evaluation
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
result=confusion_matrix(y_test,y_pred)
result


# In[18]:


#accuracy score 
score=accuracy_score(y_test,y_pred)
score


# In[ ]:




