#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler


# In[2]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# In[3]:


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[4]:


strain=pd.read_csv("D:\\data science\\assignments\\ass-17 support vector machines\\SalaryData_Train(1).csv")
stest=pd.read_csv("D:\\data science\\assignments\\ass-17 support vector machines\\SalaryData_Test(1).csv")


# In[5]:


strain


# In[6]:


strain.info


# In[7]:


strain.shape


# In[8]:


strain.head


# In[9]:


stest


# In[10]:


stest.info


# In[11]:


stest.shape


# In[12]:


stest.head


# In[13]:


strain.columns
stest.columns
stringcol=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']


# In[14]:


from sklearn import preprocessing


# In[15]:


label_encoder=preprocessing.LabelEncoder()
for i in stringcol:
    strain[i]=label_encoder.fit_transform(strain[i])
    stest[i]=label_encoder.fit_transform(stest[i])


# In[16]:


strain.head()


# In[17]:


#coverting Y column in train test both 
strain['Salary'] = label_encoder.fit_transform(strain['Salary'])


# In[18]:


stest['Salary'] = label_encoder.fit_transform(stest['Salary'])


# In[19]:


strain


# In[20]:


stest


# In[21]:


strainx=strain.iloc[:,0:13]
strainy=strain.iloc[:,13]
stestx=stest.iloc[:,0:13]
stesty=stest.iloc[:,13]


# In[22]:


strainx.shape ,strainy.shape, stestx.shape, stesty.shape,


# In[23]:


model_rbf=SVC(kernel='rbf')


# In[24]:


model_rbf.fit(strainx,strainy)


# In[25]:


train_pred_rbf=model_rbf.predict(strainx)
test_pred_rbf=model_rbf.predict(stestx)


# In[34]:


#train_rbf_acc


# In[33]:


#test_rbf_acc 


# # 2nd DataSet

# In[35]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# In[36]:


from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# In[37]:


from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score


# In[38]:


fs=pd.read_csv("D:\\data science\\assignments\\ass-17 support vector machines\\forestfires.csv")
fs


# In[40]:


fs.head()


# In[41]:


from sklearn import preprocessing

label_encoder=preprocessing.LabelEncoder()
fs['month'] = label_encoder.fit_transform(fs['month'])
fs['day'] = label_encoder.fit_transform(fs['day'])
fs['size_category'] = label_encoder.fit_transform(fs['size_category'])


# In[42]:


fs


# In[43]:


fs=fs.drop(fs.columns[0:2],axis=1)
fs.head()


# In[44]:


fs.info()


# # Splitting the data into x and y as input and output# Splitting the data into x and y as input and output

# In[45]:


x= fs.iloc[:,0:28]
y = fs.iloc[:,28]


# In[47]:


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
array_x=scaler.fit_transform(x)


# In[48]:


array_x


# In[49]:


x=pd.DataFrame(array_x)
x


# In[50]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)


# In[51]:


clf = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[60,50,5,10,0.5],'C':[20,15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(clf,param_grid,cv=10)
gsv.fit(x_train,y_train)


# In[52]:


gsv.best_params_ , gsv.best_score_ 


# In[53]:


clf = SVC(C= 20, gamma = 60)
clf.fit(x_train , y_train)
y_pred = clf.predict(x_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)


# In[ ]:




