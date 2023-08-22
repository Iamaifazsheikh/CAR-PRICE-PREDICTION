import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv('CarPrice_Assignment.csv')
df.dtypes
df.describe()
df.corr() df = df.dropna()
df = df[df['price'] < 100000] 
df['price/hp'] = df['price']/df['horsepower'] 
sns.regplot(x='horsepower', y='price', data=df)
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred) 
print("MSE:", mse)
print("R-squared:", lr.score(X_test, y_test))
