import numpy as np
from utils import split_data_into_training_and_testing_sets


def linear_regression(X, Y):
    """Linear regression algorithm."""
    # Split the data based on a desired ratio, e.g., 80% for training and 20% for testing
    X_train, X_test, Y_train, Y_test = split_data_into_training_and_testing_sets(X, Y)

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

    return intercept, coefficient_values, X_test, Y_test, Y_pred

def apply_coeficients(X, coeficients):
    return X @ coeficients