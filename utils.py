import csv
from typing import Callable, Tuple, Dict

def load_file(file_name: str) -> list:
    """Load file into memory."""
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    return data

def map_data_to_type(data: list, columns_map: Dict[str, Tuple[str, any]]) -> list:
    """Map data to type."""

    headers_list = data.pop(0)
    for header in headers_list:
        if header not in columns_map.keys():
            raise Exception('Invalid headers: {}'.format(header))

    data_list = []
    for data_row in data:
        data_list.append({
            header_name: header_type(data_row[headers_list.index(header_label)])
            for header_label, (header_name, header_type) in columns_map.items()
        })

    return data_list

def filter_data(data: list, filter: Dict[str, Callable[[str], bool]]) -> list:
    """Filter data."""
    
    for key, _ in filter.items():
        if len(data) and key not in data[0].keys():
            raise Exception('Invalid filter: {}'.format(key))

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
                raise Exception('Invalid data label: {}'.format(data_label))

    data_list = []

    for data_row in data:
        data_list.append([
            data_row[data_label]
            for data_label in row_data_order
        ])

    return data_list
