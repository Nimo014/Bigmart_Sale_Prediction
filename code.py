import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error 

df = pd.read_csv('Boston_House.csv')

#PREPROCESSING
cols = ['crim','zn','tax','b']

#minmax normalization
'''
for col in cols:
    minimum = min(df[col])
    maximum = max(df[col])
    df[col] = (df[col]-minimum)/(maximum-minimum)
'''

#standardization
for col in cols:
    x = np.mean(df[col])
    sd = np.std(df[col])
    df[col] = (df[col]-x)/sd
'''
fig,ax = plt.subplots(7,2,figsize=(10,20))

ax=ax.flatten()
i=0
for col in df:
    sns.distplot(df[col],ax = ax[i])
    i+=1
plt.tight_layout(pad=0.5,w_pad=0.7,h_pad =5)
corr =  df.corr()
plt.figure(figsize=(10,20))
sns.heatmap(corr,annot = True,cmap = 'coolwarm')
'''

#TRAINING DATA

x = df.drop(columns = ['rad','medv'])
y = df['medv']

def train(model,x,y):
    model.fit(x,y)
    y_pred = model.predict(x)

    CV = cross_val_score(model,x,y,scoring = 'neg_mean_squared_error',cv=5)
    
    print(f"MSE:{mean_squared_error(y,y_pred)}")
    print(f"CVS:{abs(np.mean(CV))}")

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor

model = ExtraTreesRegressor()
train(model,x,y)

coef = pd.Series(model.feature_importances_ , x.columns).sort_values()
coef.plot(kind = 'bar',title = 'feature importances')
plt.xticks(rotation=0)
plt.show()
