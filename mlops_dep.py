#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd


# In[56]:


df9=pd.read_csv("titanic.csv")
df9


# In[58]:


df10=df9.drop(['Embarked', 'Cabin', 'Ticket', 'Parch', 'Name', 'PassengerId'], axis='columns')
df10


# In[60]:


dummies= pd.get_dummies(df10, columns=['Sex'])
dummies


# In[62]:


from sklearn import tree


# In[64]:


model=tree.DecisionTreeClassifier()


# In[66]:


from sklearn.model_selection import train_test_split


# In[68]:


x=dummies.drop(['Survived'], axis='columns')
x


# In[70]:


y=dummies['Survived']
y


# In[72]:


x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2)


# In[74]:


model.fit(x_train, y_train)


# In[76]:


model.predict(x_test)


# In[78]:


model.score(x_test, y_test)


# In[80]:


import numpy as np


# In[82]:


def predict_survival(Pclass,Age,SibSp,Fare,Sex_female, Sex_male):
    X=np.zeros(len(x.columns))
    X[0]= Pclass
    X[1]= Age
    X[2]= SibSp
    X[3]= Fare
    X[4]= Sex_female
    X[5]= Sex_male
    return model.predict([X])[0]


# In[84]:


x.head()


# In[86]:


predict_survival(1, 50.0, 0, 53.0,1,0)


# In[88]:


import pickle
with open('titanic_model.pickle', 'wb') as f:
    pickle.dump(model, f)


# In[90]:


import json
columns = {
    'data_columns' : [col.lower() for col in x.columns]
}
with open('columns.json', "w") as f:
    f.write(json.dumps(columns))


# In[92]:


get_ipython().system('pip install flask')


# In[ ]:


get_ipython().run_line_magic('run', 'server1.py')


# In[ ]:




