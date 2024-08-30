import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load the data
# Assume a CSV file with columns: 'Genre', 'Director', 'Actors', 'Rating'
data = pd.read_csv('movie_data.csv')

# Step 2: Data Preprocessing
# Handle missing values (if any)
data.dropna(inplace=True)

# Encode categorical variables
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(data[['Genre', 'Director', 'Actors']])

# Combine encoded features with numerical columns
X = np.hstack((encoded_features, data[['Budget', 'Runtime']].values))
y = data['Rating'].values

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train the model using Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions and evaluate the model
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R² Score: {r2}")

# Example of predicting rating for a new movie
new_movie = [['Action', 'Christopher Nolan', 'Leonardo DiCaprio']]
encoded_new_movie = encoder.transform(new_movie)
additional_features = np.array([[150000000, 120]])  # Budget and Runtime
new_movie_features = np.hstack((encoded_new_movie, additional_features))

predicted_rating = model.predict(new_movie_features)
print(f"Predicted Rating: {predicted_rating[0]}")
