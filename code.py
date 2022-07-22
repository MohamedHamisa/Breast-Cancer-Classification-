# !pip install pycaret # It is an end-to-end machine learning and model management tool that speeds up the experiment cycle exponentially and makes you more productive

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pycaret.classification import *
%matplotlib inline
warnings.filterwarnings('ignore')

df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df.head()

# delete unnecessary columns
df = df.drop(columns=['id', 'Unnamed: 32'], axis=1)

# statistical info
df.describe()

# datatype info
df.info()

sns.countplot(df['diagnosis'])

df_temp = df.drop(columns=['diagnosis'], axis=1)

# create dist plot
fig, ax = plt.subplots(ncols=6, nrows=5, figsize=(20, 20))
index = 0
ax = ax.flatten()

for col in df_temp.columns:
    sns.distplot(df[col], ax=ax[index])
    index+=1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)

# create box plot
fig, ax = plt.subplots(ncols=6, nrows=5, figsize=(20, 20))
index = 0
ax = ax.flatten()

for col in df_temp.columns:
    sns.boxplot(y=col, data=df, ax=ax[index])
    index+=1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


#Create and Train the Model
# setup the data
clf = setup(df, target='diagnosis') #classifier

# train and test the models
compare_models()

# select the best model
model = create_model('catboost')

# hyperparameter tuning
best_model = tune_model(model)

evaluate_model(best_model)

# plot the results
plot_model(estimator=best_model, plot='confusion_matrix')
