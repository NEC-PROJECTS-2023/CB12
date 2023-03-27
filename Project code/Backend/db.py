import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# loading the csv data to a Pandas DataFrame
gold_data = pd.read_csv('gold price prediction.csv')
gold_data.describe()
correlation = gold_data.corr()
X = gold_data.drop(['Date','GLD'],axis=1)
Y = gold_data['EUR/USD']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)
regressor = RandomForestRegressor(n_estimators=100)
# training the model
regressor.fit(X_train,Y_train)

pickle.dump(regressor,open('gold.pkl','wb'))