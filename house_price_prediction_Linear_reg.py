import pandas as pd
import numpy as np
import pickle

# Loading the Dataset
df = pd.read_csv(r"C:\Users\dilli\Machine Learning\House Price Prediction\USA_Housing.csv")
X = df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]
y = df['Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train =  sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creating the Model
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)

# Creating a pickle file for the classifier
filename = 'house_price_prediction_liner_reg.pkl'
pickle.dump(regression, open(filename, 'wb'))
