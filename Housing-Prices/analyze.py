import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Acquire the data and create our environment
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Processing and visualising
train.SalePrice.describe()
plt.hist(train.SalePrice, color="blue") #histogram, + skewed

print ("Skew is: ", (np.log(train.SalePrice)).skew()) # applying log to skewed
plt.hist(np.log(train.SalePrice), color="blue") # removed skewness, visual

numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes

plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))





