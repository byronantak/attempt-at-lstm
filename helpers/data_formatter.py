from typing import List
import numpy as np
import pandas as pd
from pandas import DataFrame

pd.options.mode.chained_assignment = None


class DataFormatter:
    @staticmethod
    def convert_to_int_array(my_values: List) -> List:
        for i in range(len(my_values)):
            value_arr = my_values[i]
            for j in range(len(value_arr)):
                try:
                    value_arr[j] = float(value_arr[j])
                except ValueError:
                    value_arr[j] = np.nan
            my_values[i] = value_arr
        return my_values

    @staticmethod
    def merge_meta_data_with_actual_data(data, meta_data):
        meta_data_to_keep = ['location', 'km2', 'popn', 'loc_altitude']
        meta_data_results = meta_data[meta_data_to_keep]
        new_data = pd.merge(data, meta_data_results, left_on='location', right_on='location', how='left')
        new_data.drop('location', axis=1)
        return new_data

    @staticmethod
    def fill_nans(data, value):

        for i in range(len(data)):
            value_arr = DataFrame(data[i])
            value_arr = value_arr.fillna(value)
            data[i] = value_arr
        return data

    @staticmethod
    def flatten_structure_to_one_dim_structure(dimension):
        temp_array = []
        for i, label in enumerate(dimension):
            if isinstance(dimension[i], pd.DataFrame):
                values_1d = dimension[i].to_numpy(dtype=float).flatten()
                temp_array = temp_array + values_1d.tolist()
        return temp_array
