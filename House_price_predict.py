import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('Housepricedata.csv');
X = dataset[['Location','Bedrooms','Bathrooms','Size','Price/SQ.Ft','Status']].values
y = dataset.iloc[:,2].values

#Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:, 0])
X[:,5] = labelencoder.fit_transform(X[:, 5])
dataset =pd.DataFrame(X)
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#Spliting the dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_Y = StandardScaler()
y_train = sc_Y.fit_transform(y_train)
y_test = sc_Y.transform(y_test)

#fitting the model Random forest regressor 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 41, random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)    

#calculate score     
from sklearn.metrics import r2_score
acc = r2_score(y_test, y_pred)*100
print("\nAccuracy - ",acc)

