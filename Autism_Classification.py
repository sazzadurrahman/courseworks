
# coding: utf-8

# ### Machine Learning (SVM, Logistic Regression, Randon Forest)
# ### Autism classification

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[48]:



print("Setup starts...")
# Classification

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier 

# Regression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


# Modelling Helpers :
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV , KFold , cross_val_score, ShuffleSplit, cross_validate
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score

# Preprocessing :
from sklearn.preprocessing import MinMaxScaler , StandardScaler, Imputer, LabelEncoder

# Metrics :
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error 
# Classification
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, classification_report

print("Setup completed...")


# In[62]:


asd = pd.read_csv("data/Toddler Autism dataset July 2018.csv")
print("Dataset loaded...")


# In[63]:


asd.describe()


# In[64]:


asd.columns


# In[69]:


asd.shape


# In[39]:


asd.drop('Qchat-10-Score', axis = 1, inplace = True)
asd.drop(['Case_No', 'Who completed the test'], axis = 1, inplace = True)


# In[67]:


plt.figure(figsize = (16,8))
sns.countplot(x = 'Ethnicity', data = asd, order = asd['Ethnicity'].value_counts().index)


# In[41]:


le = LabelEncoder()
columns = ['Ethnicity', 'Family_mem_with_ASD', 'Class/ASD Traits ', 'Sex', 'Jaundice']
for col in columns:
    asd[col] = le.fit_transform(asd[col])
asd.dtypes


# In[42]:


X = asd.drop(['Class/ASD Traits '], axis = 1)
Y = asd['Class/ASD Traits ']


# In[79]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 7)


# In[80]:


models = []
models.append(('Support Vector Machine:      ', SVC()))
models.append(('Random Forest:               ', RandomForestRegressor()))
models.append(('Logistic Regression:         ', LogisticRegression()))

for name, model in models:
    model.fit(x_train, y_train)
    pred = model.predict(x_test).astype(int)
    print(name, accuracy_score(y_test, pred))


# In[83]:


param_grid = [{'C': np.logspace(-3, 3, 10)}]

grid_search_SVC = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42),
    scoring='f1',
    n_jobs=-1
)

scores_SVC = cross_val_score(
    estimator=grid_search_SVC,
    X=x_test,
    y=y_test,
    cv=StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0),
    scoring='f1',
    n_jobs=-1
)


# In[84]:


scores_SVC.mean()*100


# In[56]:


#Fine tuning SVC parameters
svc = SVC()

params = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}

#print('SVC paramters tuning starts...')
clf = GridSearchCV(svc, param_grid = params, scoring = 'accuracy', cv = 10, verbose = 2)

clf.fit(x_train, y_train)
clf.best_params_
#print('SVC paramters tuning finished...')


# In[57]:


# Re-running model with best parametres
svc1 = SVC(C = 0.8, gamma = 0.1, kernel = 'linear')
svc1.fit(x_train, y_train)
pred = svc1.predict(x_test)
print(accuracy_score(y_test, pred))


# In[87]:


#Fine tuning Random Forest Classifier 
rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[88]:


CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(x_train, y_train)
CV_rfc.best_params_


# In[89]:


rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 200, max_depth=8, criterion='gini')
rfc1.fit(x_train, y_train)


# In[90]:


pred=rfc1.predict(x_test)
print("Accuracy for Random Forest on CV data: ",accuracy_score(y_test,pred))


# In[51]:


#Logistic Regression
print ('Tuning starts...')
param_grid = [{'C': np.logspace(-3, 3, 10)}]

grid_search = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid=param_grid,
    cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42),
    scoring='f1',
    n_jobs=-1
)

scores = cross_val_score(
    estimator=grid_search,
    X=x_test,
    y=y_test,
    cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0),
    scoring='f1',
    n_jobs=-1
)
print ('Tuning finished...')


# In[52]:


scores.mean()*100

