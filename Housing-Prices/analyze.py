import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Acquire the data and create our environment
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Processing and visualising
train.SalePrice.describe()
plt.hist(train.SalePrice, color="blue") #histogram, + skewed





