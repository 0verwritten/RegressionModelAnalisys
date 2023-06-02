from utils import *
from linear_regression import linear_regression
from sklearn.linear_model import LinearRegression
import pandas as pd

COLUMNS_MAP = {
    "CustomerID": ("customer_id", int),
    "Gender": ("gender", str),
    "Age": ("age", int),
    "Annual Income ($)": ("annual_income", int),
    "Spending Score (1-100)": ("spending_score", int),
    "Profession": ("profession", str),
    "Work Experience": ("work_experience", int),
    "Family Size": ("family_size", int),
}

customer_data = map_data_to_type(load_file("data.csv"), COLUMNS_MAP)
# print(filter_data(customer_data, {'age': lambda x: True})[0])

X = prepate_data_to_regresson(
    customer_data, ('age',)
)
Y = flatten_list(prepate_data_to_regresson(customer_data, ("spending_score",)))

# X, Y = sort_input_data(X, Y)

# print(*zip(X,Y), sep='\n')

intercept, coefficient_values, X_test, Y_test, Y_pred = linear_regression(X, Y)
print(coefficient_values)
evaluate_model(Y_test, Y_pred)
print(Y_test)
print('regression line', linear_regression(range(len(Y)), Y))

# for x in range(len(X_test[0]))
print_model(intercept, coefficient_values,  (range(len(Y_test)), Y_test), (range(len(Y_test)), Y_pred))
