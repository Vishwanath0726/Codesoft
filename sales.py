import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import csv

# Step 1: Load the data (Replace this part with actual data loading)
# Assume we have a CSV file with columns: TV, Radio, Social Media, Sales
data = []
with open('advertising_data.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        data.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])

data = np.array(data)
X = data[:, :-1]  # Features: TV, Radio, Social Media
y = data[:, -1]   # Target: Sales

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train the model using Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions and evaluate the model
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")

# Visualize the results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Example of predicting sales for a new advertising expenditure
new_data = np.array([[230.1, 37.8, 69.2]])  # Example: TV, Radio, Social Media spends
predicted_sales = model.predict(new_data)
print(f"Predicted Sales: {predicted_sales[0]}")
