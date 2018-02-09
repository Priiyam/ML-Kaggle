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

# Outliers can affect a regression model by pulling our estimated regression line further away from 
# the true population regression line. So, we'll remove those observations from our data. Removing outliers is 
# an art and a science. 

# Removing outlines for GarageArea
train = train[train['GarageArea'] < 1200]

# Checking null values
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ["Null Count"]
nulls.index.name = "Feature"

# Unique values
print ("Unique values are:", train.MiscFeature.unique())





