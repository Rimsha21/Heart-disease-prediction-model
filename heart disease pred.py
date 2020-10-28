
# coding: utf-8

# # HEART DISEASE DETECTION MODEL

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[6]:


df=pd.read_csv("heart.csv")


# In[7]:


df.head()


# In[10]:


from sklearn.neighbors import KNeighborsClassifier


# In[11]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[12]:


df.info()


# In[13]:


df.describe()


# In[14]:


df.corr()


# In[15]:


sns.countplot(x='target',data=df)


# In[19]:


dataset= pd.get_dummies(df, columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])


# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[21]:


sc= StandardScaler()


# In[22]:


columns_to_scale= ['age','trestbps','chol','thalach','oldpeak']
dataset[columns_to_scale]=sc.fit_transform(dataset[columns_to_scale])


# In[23]:


dataset.head()


# In[76]:


dataset['target']=df.target
dataset['target'].head()
x=dataset.drop(['target'],axis=1)
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_test.shape
knn_classifier=KNeighborsClassifier(n_neighbors=12)
knn_classifier.fit(x_train,y_train)
y_pred_knn=knn_classifier.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred_knn)


# # RANDOM FOREST

# In[90]:


rf_classifier= RandomForestClassifier(n_estimators=300,criterion='entropy',min_samples_split=10)
rf_classifier.fit(x_train,y_train)
y_pred_rf=rf_classifier.predict(x_test)
accuracy_score(y_test,y_pred_rf)


# # decision tree classifier

# In[95]:


dt_classifier=DecisionTreeClassifier()
dt_classifier.fit(x_train,y_train)
y_pred_dt=dt_classifier.predict(x_test)
accuracy_score(y_test,y_pred_dt)

