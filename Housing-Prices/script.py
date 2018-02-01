import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
X = data.drop(['SalePrice', 'Id'], axis=1)
y = np.log(data.SalePrice)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)

from sklearn.linear_model import LinearRegression #0.02192, 0.204, 
regressor = LinearRegression()
regressor.fit(X, y)

y_pred = regressor.predict(X_test)


