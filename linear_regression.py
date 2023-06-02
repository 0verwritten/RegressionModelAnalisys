import numpy as np
import matplotlib.pyplot as plt

# Step 1: Prepare the data
# Assume we have a dependent variable Y and independent variables X1, X2, X3, X4, X5
# X and Y should be numpy arrays or lists

# Assuming X and Y are already defined and preprocessed
# X = [[19, 15000, 4, 1], [21, 35000, 3, 3], [20, 86000, 1, 1], [23, 59000, 2, 0], [31, 38000, 6, 2], [22, 58000, 2, 0], [35, 31000, 3, 1], [23, 84000, 3, 1], [64, 97000, 3, 0], [30, 98000, 4, 1]]
# Y = [39, 81, 6, 77, 40, 76, 6, 94, 3, 72]

# Generate random test data
np.random.seed(42)

# Define the number of data points
num_points = 100

# Define the number of independent variables
num_features = 5

# Generate random independent variables (X) with values between 0 and 10
X = np.random.uniform(low=0, high=10, size=(num_points, num_features))

# Define the true coefficients for each independent variable
true_coefficients = np.array([2, -1, 0.5, 3, -2])

# Generate the dependent variable (Y) using the true coefficients and adding some random noise
noise = np.random.normal(loc=0, scale=1, size=num_points)

print(np.where(X == X[0]), X[1])
Y = np.dot(X, true_coefficients) + noise

X = sorted(X, key=lambda x: Y[np.where(np.isclose(X, x))[0][0]])
Y = sorted(Y)


# Step 2: Split the data into training and testing sets
# Split the data based on a desired ratio, e.g., 80% for training and 20% for testing
split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train = X[:split_index]
X_test = X[split_index:]

Y_train = Y[:split_index]
Y_test = Y[split_index:]

# Step 3: Implement the linear regression algorithm

# Add a column of ones to X_train and X_test to account for the intercept term
X_train = np.column_stack((np.ones(len(X_train)), X_train))
X_test = np.column_stack((np.ones(len(X_test)), X_test))

# Calculate the regression coefficients using the normal equation (closed-form solution)
coefficients = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ Y_train

# Extract the intercept and coefficient values
intercept = coefficients[0]
coefficient_values = coefficients[1:]

# Step 4: Make predictions on the testing data
Y_pred = X_test @ coefficients

# Step 5: Evaluate the model
# Calculate mean squared error (MSE) and R-squared value
mse = np.mean((Y_test - Y_pred) ** 2)
y_mean = np.mean(Y_test)
ss_total = np.sum((Y_test - y_mean) ** 2)
ss_residual = np.sum((Y_test - Y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)

# Step 6: Print the evaluation metrics
print("Mean Squared Error:", mse)
print("R-squared:", r2)
# print(X_test)
plt.scatter(range(len(X_test)), Y_test, color='green', label='Actual')
plt.scatter(range(len(X_test)), Y_pred, color='blue', label='Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression - Actual vs Predicted')
plt.legend()
plt.show()