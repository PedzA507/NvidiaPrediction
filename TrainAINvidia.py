import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

# Load data
data = pd.read_csv('Nvidia.csv')

# Shift 'Close' to use the next day's close price as the target
data['Target_Close'] = data['Close'].shift(-1)

# Drop the last row with NaN due to shift operation
data = data.dropna()

# Define features and target
x = data.drop(columns=['Date', 'Close', 'Target_Close'])  # Use all columns except 'Date', 'Close', and the shifted target
y = data['Target_Close']  # Use 'Target_Close' as the target

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Use GridSearchCV to find the best parameters for RandomForestRegressor
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, None]}
rf = RandomForestRegressor(random_state=0)
rf_gscv = GridSearchCV(rf, param_grid, cv=5)

# Train the model
rf_gscv.fit(x_train, y_train)

# Get the best parameters
best_params = rf_gscv.best_params_
best_rf = RandomForestRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], random_state=0)
best_rf.fit(x_train, y_train)

# Make predictions
y_pred = best_rf.predict(x_test)

# Evaluate the model
mape = np.mean(np.absolute(y_test - y_pred) / y_test * 100)

# Save the model
joblib.dump(best_rf, "Nvidia.pkl")

print(f'Best parameters: {best_params}, MAPE: {np.round(mape, 2)}')
