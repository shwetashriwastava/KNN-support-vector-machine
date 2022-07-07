#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


zoo=pd.read_csv("D:\\data science\\assignments\\ass-13 KNN\\Zoo.csv")
zoo


# In[5]:


zoo.isnull().sum()


# In[6]:


zoo.info


# In[7]:


array = zoo.values
X= zoo.iloc[:,1:17]
Y=zoo.iloc[:,17]


# In[8]:


array


# In[10]:


X


# In[11]:


Y


# # Splitting data into training and testing data set
# 

# In[12]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.30, random_state=40)


# # KNN Classification

# In[13]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[14]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)


# In[15]:


y_pred = classifier.predict(x_test)


# In[16]:


y_test


# In[17]:


y_pred


# In[18]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)


# In[19]:


print(cm)


# In[20]:


print(ac)


# In[21]:


n_neighbors = np.array(range(1,50))
param_grid = dict(n_neighbors=n_neighbors)


# In[22]:


model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)


# In[23]:


print(grid.best_score_)
print(grid.best_params_)


# In[24]:


k_range = range(1, 4)
k_scores = []


# In[25]:


for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=2)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# # assignment (Glass) KNN

# In[26]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[27]:


glass=pd.read_csv("D:\\data science\\assignments\\ass-13 KNN\\glass.csv")


# In[28]:


glass


# In[29]:


glass.info()


# In[30]:


glass.isnull().sum()


# In[31]:


array = glass.values
X= glass.iloc[:,0:9]
Y=glass.iloc[:,9]


# In[32]:


X


# In[33]:


Y


# In[34]:


num_folds = 10
kfold = KFold(n_splits=10)


# In[35]:


model = KNeighborsClassifier(n_neighbors=17)
results = cross_val_score(model, X, Y, cv=kfold)


# In[36]:


print(results.mean())


# In[37]:


from sklearn.model_selection import GridSearchCV


# In[38]:


n_neighbors = np.array(range(1,170))
param_grid = dict(n_neighbors=n_neighbors)
model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)


# In[39]:


print(grid.best_score_)
print(grid.best_params_)


# In[ ]:





# In[ ]:





# In[ ]:




