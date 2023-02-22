#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np


# In[19]:


df=pd.read_csv(r"C:\Users\ANJANA\Desktop\ML\lung_cancer_examples.csv")
df.head()


# In[20]:


df.shape


# In[21]:


df.describe(include='all')


# In[22]:


df.isna().sum()


# In[23]:


df["Result"].value_counts()


# In[24]:


df1=df.drop(['Name','Surname'],axis=1)
df1


# In[25]:


x=df1.iloc[:,:-1].values
x


# In[26]:


y=df1.iloc[:,-1].values
y


# In[27]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=1)
x_train


# In[28]:


x_test


# In[29]:


y_train


# In[30]:


y_test


# In[31]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x_train
x_test


# In[32]:


#SVM
from sklearn.svm import SVC
classifier=SVC()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
y_pred


# In[33]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,ConfusionMatrixDisplay
label=[0,1]
print(classification_report(y_test,y_pred))
matrix=confusion_matrix(y_test,y_pred)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
cmd=ConfusionMatrixDisplay(matrix,display_labels=label)
cmd.plot()


# In[34]:


#KNN Algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y_train)


# In[35]:


y_pred=classifier.predict(x_test)
y_pred


# In[36]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
result=confusion_matrix(y_test,y_pred)
result


# In[38]:


#Naive bayes 
from sklearn.naive_bayes import GaussianNB
clfr=GaussianNB()
clfr.fit(x_train,y_train)
y_pred=clfr.predict(x_test)
y_pred


# In[39]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,ConfusionMatrixDisplay
report=classification_report(y_test,y_pred)
report


# In[43]:


from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay
label=[0,1]
result=confusion_matrix(y_test,y_pred)
result
cmd=ConfusionMatrixDisplay(result,display_labels=label)
cmd.plot()


# In[44]:


score=accuracy_score(y_test,y_pred)
score

