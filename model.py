# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 02:48:05 2020

@author: Bhavdeep
"""

# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv("dataset/insurance.csv")
df


# # Data Analysis

# ### 1) Missing Values

# In[ ]:


f_na = [feature for feature in df.columns if df[feature].isna().sum()>0]
f_na


# * No missing values in features

# ### 2) Numerical Features

# In[ ]:


f_num = [feature for feature in df.columns if df[feature].dtype!='object']
f_num


# In[ ]:


# Discrete Numerical Features
f_dis = [feature for feature in f_num if len(df[feature].unique())<25]
print(f_dis)
# Continuous Numerical Features
f_cont = [feature for feature in f_num if feature not in f_dis]
print(f_cont)


# In[ ]:


import matplotlib.pyplot as plt
for feature in f_cont:
    plt.hist(df[feature], bins=50)
    plt.title(feature)
    plt.show()


# In[ ]:


for feature in f_cont:
    data = df.copy()
    data[feature] = np.log(data[feature])
    plt.hist(data[feature], bins=50)
    plt.title(feature)
    plt.show()


# In[ ]:


import statsmodels.api as sm
import scipy.stats as stats
#import pylab 
fig, axs = plt.subplots(len(f_cont), 2, figsize=(15,15))

for i in range(len(f_cont)):
    data = df.copy()
    data[f_cont[i]] = np.sqrt(data[f_cont[i]])
    stats.probplot(df[f_cont[i]], dist="norm", plot=axs[i,0])
    axs[i,0].set_title(f_cont[i])
    stats.probplot(data[f_cont[i]], dist="norm", plot=axs[i,1])
    axs[i,1].set_title(f_cont[i])
    
    #plt.title(feature)
plt.show()

# * Taking sqrt of 'age' feature will make it gaussian distributed

# In[ ]:

# ### 3) Categorical Features

# In[ ]:


f_cat = [feature for feature in df.columns if df[feature].dtype=='O']
f_cat


# In[ ]:


for i in f_cat:
    print(df[i].unique(),"\n",df[i].value_counts())


# ### 4) Outliers

# In[ ]:


import seaborn as sns
sns.set(style="whitegrid")
for i in f_num:
    ax = sns.boxplot(x=df[i])
    plt.show()

# In[ ]:


from scipy import stats 
IQR=[]
for i in f_num:
    IQR.append(stats.iqr(df[i], interpolation = 'midpoint'))
print(IQR)


# In[ ]:


limits = dict()
j=0
for i in f_num:
    Q1 = np.percentile(df[i], 25, interpolation = 'midpoint')  
    Q3 = np.percentile(df[i], 75, interpolation = 'midpoint')  
    #print(Q1, Q3)
    limits[i] = [Q1-(1.5*IQR[j]), Q3+(1.5*IQR[j])]
    j+=1

print(limits)


# In[ ]:


outliers = dict()
for i in f_num:
    outliers[i]=list()
    for x in df[i]:
        if(x<limits[i][0] or x>limits[i][1]):
            outliers[i].append(x)
   
print(outliers)



# # Feature Engineering

# In[ ]:


"""

# One Hot Encoding for 'sex' and 'smoker' columns
df_new = df.copy()
pd.get_dummies(df_new, columns=['sex','smoker'], prefix=['sex', 'smoker'])


# Binary Encoding for 'sex' and 'smoker' columns
import category_encoders as ce
df_new = df.copy()
encoder = ce.BinaryEncoder(cols=['sex','smoker'])
df_new = encoder.fit_transform(df_new)
print(df_new.head())

"""


# ### Removing Outliers

# In[ ]:


#df_new = df.drop(df[df['bmi']>=47].index, axis=0).reset_index(drop=True)

df_new = df.copy()


# ### Handling Categorical Features

# In[ ]:


"""
# Backward Difference Encoding for 'region' column
import category_encoders as ce
encoder = ce.BackwardDifferenceEncoder(cols=['region'])
df_new = encoder.fit_transform(df_new)
print(df_new.head())
"""

# One Hot Encoding for 'region' column
import category_encoders as ce
import pandas as pd

#Create object for one-hot encoding
encoder=ce.OneHotEncoder(cols=['region'],handle_unknown='return_nan',return_df=True,use_cat_names=True)

#Original Data
print(df_new.head())
#Fit and transform Data
df_new = encoder.fit_transform(df_new)
print(df_new.head())


# Label Encoding for 'sex' and 'smoker' columns
from sklearn.preprocessing import LabelEncoder
df_new['sex'] = LabelEncoder().fit_transform(df_new['sex'])
df_new['smoker'] = LabelEncoder().fit_transform(df_new['smoker'])
df_new.head()


# In[ ]:

# ### SQRT Transformation

# In[ ]:

df_new['age'] = np.sqrt(df_new['age'])
df_new

# # Feature Scaling

# In[ ]:


"""
## StandardScaler
from sklearn.preprocessing import StandardScaler
data = df_new.copy()
scaler = StandardScaler()
data[data.drop(['charges'],axis=1).columns] = scaler.fit_transform(data.drop(['charges'],axis=1))
data = pd.DataFrame(data, columns=df_new.columns)
#print(scaler.mean_)
#print(scaler.transform([[2, 2]]))

## MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
data = df_new.copy()
scaler = MinMaxScaler()
data[data.drop(['charges'],axis=1).columns] = scaler.fit_transform(data.drop(['charges'],axis=1))
data = pd.DataFrame(data, columns=df_new.columns)

## MaxAbsScaler
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
data = df_new.copy()
data[data.drop(['charges'],axis=1).columns] = scaler.fit_transform(data.drop(['charges'],axis=1))
data = pd.DataFrame(data, columns=df_new.columns)

## RobustScaler
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
data = df_new.copy()
data[data.drop(['charges'],axis=1).columns] = scaler.fit_transform(data.drop(['charges'],axis=1))
data = pd.DataFrame(data, columns=df_new.columns)

## QuantileTransformer
from sklearn.preprocessing import QuantileTransformer
scaler = QuantileTransformer()
data = df_new.copy()
data[data.drop(['charges'],axis=1).columns] = scaler.fit_transform(data.drop(['charges'],axis=1))
data = pd.DataFrame(data, columns=df_new.columns)

## PowerTransformer using yeo-johnson
from sklearn.preprocessing import PowerTransformer
scaler = PowerTransformer(method='yeo-johnson')
data = df_new.copy()
data[data.drop(['charges'],axis=1).columns] = scaler.fit_transform(data.drop(['charges'],axis=1))
data = pd.DataFrame(data, columns=df_new.columns)

## PowerTransformer using box-cox
from sklearn.preprocessing import PowerTransformer
scaler = PowerTransformer(method='box-cox')
data = df_new.copy()
data[data.drop(['charges'],axis=1).columns] = scaler.fit_transform(data.drop(['charges'],axis=1))
data = pd.DataFrame(data, columns=df_new.columns)
"""


# In[ ]:


data = df_new.copy()


# In[ ]:


plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), annot=True)


# In[ ]:

X, y = data.iloc[:,:-1], data.iloc[:,-1]

# # Feature Selection

# In[ ]:



"""
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, y)

from sklearn.feature_selection import RFE
selector = RFE(reg, n_features_to_select=5, step=1)
selector = selector.fit(X, y)
print(selector.support_)
print(selector.ranking_)

f_selected = X.columns[selector.support_]
f_selected

"""


# In[ ]:


# Feature Importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesRegressor

# feature extraction
model = ExtraTreesRegressor(n_estimators=10)
model.fit(X, y)
print(model.feature_importances_)
top_features = np.array(model.feature_importances_)

indices = (-top_features).argsort()[:3]
print(indices)
f_selected = X.columns[indices]
print(X.columns)
print(f_selected)


# # Train-Test Split

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=128)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


X_train = X_train[f_selected]
X_test = X_test[f_selected]
print(X_train.shape)
print(X_test.shape)


# In[]

# # Model Building

# #### We selected GradientBoostingRegressor

# # Hyperparameter Optimization

# ### 1) RandomizedSearchCV

# In[ ]:
"""

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 300)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [int(x) for x in np.linspace(2, 10)]
# Minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in np.linspace(2, 10)]
loss = ['ls', 'lad', 'huber', 'quantile']
learning_rate = [0.01, 0.03, 0.1, 0.3]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'loss' : loss,
               'learning_rate' : learning_rate,
               }


# In[ ]:

from sklearn.ensemble import GradientBoostingRegressor
# Use the random grid to search for best hyperparameters
# First create the base model to tune
gradientboost_reg = GradientBoostingRegressor(random_state=0)
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
gb_random = RandomizedSearchCV(estimator = gradientboost_reg, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1)
# Fit the random search model
gb_random.fit(X_train[f_selected], y_train)


# In[ ]:


print(gb_random.best_params_)
print(gb_random.best_estimator_)


# In[ ]:


best_random = gb_random.best_estimator_

#gradientboost_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, max_features='sqrt', random_state=0)
best_random.fit(X_train[f_selected], y_train)

from sklearn.metrics import mean_absolute_error
y_pred = best_random.predict(X_train[f_selected])
y_test_pred = best_random.predict(X_test[f_selected])
mae_train = mean_absolute_error(y_train, y_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
print("MAE for train :",mae_train)
print("MAE for test :",mae_test)

from sklearn.metrics import r2_score
print("R2 for train :",r2_score(y_train, y_pred))
print("R2 for test :",r2_score(y_test, y_test_pred))
#print(score.mean())

"""

# ### 2) Bayesian Optimization

# ### GradientBoostingRegressor

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

def black_box_function(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    
    n_estimators = int(n_estimators*250 + 50)
    max_depth = int(max_depth*8 + 2)
    min_samples_split = int(min_samples_split*8 + 2)
    min_samples_leaf = int(min_samples_leaf*8 + 2)
    
    gradientboost_reg = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,                                                  min_samples_split=min_samples_split,                                                  min_samples_leaf=min_samples_leaf, random_state=0)
    score = cross_val_score(gradientboost_reg, X_train[f_selected], y_train, cv=10)
    r2 = score.mean()
    n = len(X_train[f_selected])
    p = X_train[f_selected].shape[1]
    
    adj_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    
    return adj_r2
    
# Number of trees in random forest
n_estimators = (0,1)
# Number of features to consider at every split
#max_features = ['auto', 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = (0, 1)
# Minimum number of samples required to split a node
min_samples_split = (0, 1)
# Minimum number of samples required at each leaf node
min_samples_leaf = (0, 1)
#loss = ('ls', 'lad', 'huber', 'quantile')

# Create the random grid
pbounds = {'n_estimators': n_estimators,
           'max_depth': max_depth,
           'min_samples_split': min_samples_split,
           'min_samples_leaf': min_samples_leaf,
           }

from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=0,
)

optimizer.maximize(
    init_points=40,
    n_iter=50,
)

print(optimizer.max)

def black_box_function_test(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    
    n_estimators = int(n_estimators*250 + 50)
    max_depth = int(max_depth*8 + 2)
    min_samples_split = int(min_samples_split*8 + 2)
    min_samples_leaf = int(min_samples_leaf*8 + 2)
    #print(n_estimators, max_depth, min_samples_split, min_samples_leaf)
    
    gradientboost_reg = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,                                                  min_samples_split=min_samples_split,                                                  min_samples_leaf=min_samples_leaf, random_state=0)
    #print(gradientboost_reg)
    gradientboost_reg.fit(X_train[f_selected], y_train)

    #from sklearn.metrics import mean_absolute_error
    y_pred = gradientboost_reg.predict(X_train[f_selected])
    y_test_pred = gradientboost_reg.predict(X_test[f_selected])
    #mae_train = mean_absolute_error(y_train, y_pred)
    #mae_test = mean_absolute_error(y_test, y_test_pred)
    #print("MAE for train :",mae_train)
    #print("MAE for test :",mae_test)

    from sklearn.metrics import r2_score
    #print("R2 for train :",r2_score(y_train, y_pred))
    #print("R2 for test :",r2_score(y_test, y_test_pred))
    r2_train = r2_score(y_train, y_pred)
    n = len(X_train[f_selected])
    p = X_train[f_selected].shape[1]
    adj_r2_train = 1-(1-r2_train)*(n-1)/(n-p-1)
    print("Train adjusted r2 score = ",adj_r2_train)
    
    #print(gradientboost_reg)
    r2_test = r2_score(y_test, y_test_pred)
    n = len(X_test[f_selected])
    p = X_test[f_selected].shape[1]
    adj_r2_test = 1-(1-r2_test)*(n-1)/(n-p-1)

    print("Test adjusted r2 score = ",adj_r2_test)
    
    
params = optimizer.max['params']
n_estimators = params['n_estimators']
max_depth = params['max_depth']
min_samples_split = params['min_samples_split']
min_samples_leaf = params['min_samples_leaf'
                         ]
print(n_estimators, max_depth, min_samples_split, min_samples_leaf)

black_box_function_test(n_estimators, max_depth, min_samples_split, min_samples_leaf)


# In[]

n_estimators = int(n_estimators*250 + 50)
max_depth = int(max_depth*8 + 2)
min_samples_split = int(min_samples_split*8 + 2)
min_samples_leaf = int(min_samples_leaf*8 + 2)
#print(n_estimators, max_depth, min_samples_split, min_samples_leaf)

gradientboost_reg = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,                                                  min_samples_split=min_samples_split,                                                  min_samples_leaf=min_samples_leaf, random_state=0)
#print(gradientboost_reg)
gradientboost_reg.fit(X_train[f_selected], y_train)

import pickle

#pickle.dump(gradientboost_reg, open('GBR_model.pkl','wb'))
