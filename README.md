# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load Dataset: Import the California Housing dataset using fetch_california_housing() for features and house price.
2. Create Multi-target Data: Use the target price and generate a dummy "occupants" column to form a 2D output array.
3. Preprocess Data: Split the data into training and test sets, and apply feature scaling using StandardScaler.
4. Train Model: Use MultiOutputRegressor with SGDRegressor to fit the model on the training data.
5. Evaluate and Predict- Predict on test data, calculate MSE, and display predicted price and number of occupants.

## Program:
```
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Hanshika Varthini R
RegisterNumber: 212223240046

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

data = fetch_california_housing()
X = data.data
price = data.target

np.random.seed(42)
occupants = np.random.randint(1, 6, size=X.shape[0]) + X[:, data.feature_names.index("AveRooms")].astype(int)
y = np.column_stack((price, occupants))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-3)
model = MultiOutputRegressor(sgd)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

for i in range(10):
    print(f"Predicted Price: ${y_pred[i][0]*100:.2f}, Predicted Occupants: {round(y_pred[i][1])}")
```

## Output:
![Screenshot 2025-04-07 155320](https://github.com/user-attachments/assets/efd54171-7314-4712-badb-c480e7f54335)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
