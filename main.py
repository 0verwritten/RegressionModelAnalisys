from utils import *

COLUMNS_MAP = {
    'CustomerID': ('customer_id', int),
    'Gender': ('gender', str),
    'Age': ('age', int),
    'Annual Income ($)': ('annual_income', int),
    'Spending Score (1-100)': ('spending_score', int),
    'Profession': ('profession', str),
    'Work Experience': ('work_experience', int),
    'Family Size': ('family_size', int),
}

customer_data = map_data_to_type(load_file('data.csv'), COLUMNS_MAP)
# print(filter_data(customer_data, {'age': lambda x: True})[0])

print(prepate_data_to_regresson(customer_data, ('age', 'annual_income', 'family_size', 'work_experience'))[:10])
print(prepate_data_to_regresson(customer_data, ('spending_score',))[:10])