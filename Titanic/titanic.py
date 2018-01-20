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

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=0)
classifier.fit(X, y)

y_pred = classifier.predict(X)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)


