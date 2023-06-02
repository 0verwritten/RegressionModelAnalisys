import csv
import operator
import functools
import numpy as np
from math import ceil
from random import randint
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Dict


def load_file(file_name: str) -> list:
    """Load file into memory."""
    with open(file_name, newline="") as f:
        reader = csv.reader(f)
        data = list(reader)

    return data


def map_data_to_type(data: list, columns_map: Dict[str, Tuple[str, any]]) -> list:
    """Map data to type."""

    headers_list = data.pop(0)
    for header in headers_list:
        if header not in columns_map.keys():
            raise Exception("Invalid headers: {}".format(header))

    data_list = []
    for data_row in data:
        data_list.append(
            {
                header_name: header_type(data_row[headers_list.index(header_label)])
                for header_label, (header_name, header_type) in columns_map.items()
            }
        )

    return data_list


def filter_data(data: list, filter: Dict[str, Callable[[str], bool]]) -> list:
    """Filter data."""

    for key, _ in filter.items():
        if len(data) and key not in data[0].keys():
            raise Exception("Invalid filter: {}".format(key))

    filtered_data = []

    for data_row in data:
        for key, filter_function in filter.items():
            if filter_function(data_row[key]):
                filtered_data.append(data_row)
                break

    return filtered_data

def prepate_data_to_regresson(data: list, row_data_order: Tuple[str]) -> list:
    """Prepare data to regression."""

    for row_data in data:
        for data_label in row_data_order:
            if data_label not in row_data.keys():
                raise Exception("Invalid data label: {}".format(data_label))

    data_list = []

    for data_row in data:
        data_list.append([data_row[data_label] for data_label in row_data_order])

    return np.array(data_list)


def flatten_list(list: list) -> list:
    return functools.reduce(operator.iconcat, list, [])

def sort_input_data(X, Y):
    X = sorted(X, key=lambda x: Y[np.where(np.isclose(X, x))[0][0]])
    Y = sorted(Y)
    return X, Y

def split_data_into_training_and_testing_sets(X,Y):
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]

    Y_train = Y[:split_index]
    Y_test = Y[split_index:]

    return X_train, X, Y_train, Y
    # return X_train, X_test, Y_train, Y_test

def generate_test_data(
    seed=randint(0, 99999999), number_of_points=100, number_of_features=5
):
    print(f"Generating test data with seed {seed}")
    np.random.seed(seed=seed)

    # Generate random independent variables (X) with values between 0 and 10
    X = np.random.uniform(low=0, high=10, size=(number_of_points, number_of_features))

    # Define the true coefficients for each independent variable
    true_coefficients = np.array([2, -1, 0.5, 3, -2])

    # Generate the dependent variable (Y) using the true coefficients and adding some random noise
    noise = np.random.normal(loc=0, scale=2, size=number_of_points)

    Y = np.dot(X, true_coefficients) + noise

    X, Y = sort_input_data(X, Y)

    return X, Y


def evaluate_model(Y_test, Y_pred):
    mse = np.mean((Y_test - Y_pred) ** 2)
    y_mean = np.mean(Y_test)
    ss_total = np.sum((Y_test - y_mean) ** 2)
    ss_residual = np.sum((Y_test - Y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    print("Mean Squared Error:", mse)
    print("R-squared:", r2)


def print_model(intercept, coefficient_values, coords_test, coords_pred):
    print(
        "The regression model is: Y = {:.5} + {}".format(
            intercept,
            " + ".join(
                [
                    "{:.5}*X{}".format(value, index)
                    for index, value in enumerate(coefficient_values)
                ]
            ),
        )
    )
    plt.scatter(*coords_test, color="green", label="Actual")
    plt.plot(*coords_pred, color="blue", label="Predicted")
    plt.scatter(*coords_pred, color="blue", label="Predicted")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Linear Regression - Actual vs Predicted")
    plt.legend()
    plt.show()
