# Titanic Problem using ANN

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("train.csv")
X = dataset.iloc[:, [2,4,5,6,7,9]].values
y = dataset.iloc[:,1].values

pd.isnull(dataset).sum()

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder( categorical_features=[1] )
X = onehotencoder.fit_transform(X).toarray()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim=4, init='uniform', activation='relu', input_dim=7))
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
classifier.add(Dense(output_dim=1, init="uniform", activation='sigmoid'))
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
classifier.fit(X, y , batch_size=10, nb_epoch=100)

y_pred = classifier.predict(X)
y_pred = (y_pred>0.5)*1

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)

# Test Set
dataset2 = pd.read_csv("test.csv")
X2 = dataset2.iloc[:, [1,3,4,5,6,8]].values
y_id = dataset2.iloc[:, 0].values

from sklearn.preprocessing import Imputer
imputer2 = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer2 = imputer2.fit(X2[:, 2:3])
X2[:, 2:3] = imputer2.transform(X2[:, 2:3])
imputer3 = imputer2.fit(X2[:, 5:6])
X2[:, 5:6] = imputer3.transform(X2[:, 5:6])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X2 = LabelEncoder()
X2[:, 1] = labelencoder_X2.fit_transform(X2[:, 1])
onehotencoder2 = OneHotEncoder( categorical_features=[1] )
X2 = onehotencoder2.fit_transform(X2).toarray()

from sklearn.preprocessing import StandardScaler
sc2 = StandardScaler()
X2 = sc2.fit_transform(X2)

# Predicting
y_pred = classifier.predict(X2)
y_pred = (y_pred>0.3)*1
y_pred = np.insert(y_pred,0,np.arange(892,1310),axis=1)
np.savetxt('result2.csv', y_pred, delimiter=',', header="PassengerId,Survived",fmt="%i")
y_joint = np.stack((y_id,y_pred))
y_joint = np.transpose(y_joint)
df = pd.DataFrame(y_joint)
df.columns = ['PassengerId', 'Survived']
df.to_csv("result2.csv", index=False)




