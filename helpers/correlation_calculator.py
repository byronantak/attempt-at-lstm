import itertools
from scipy.stats.stats import pearsonr
import pandas as pd
import numpy as np


class CorrelationCalculator:
    @staticmethod
    def calculate_correlations(list_of_cols, df):
        dict = {'correlations': []}
        pairs_of_cols = list(itertools.combinations(list_of_cols, 2))
        for col1, col2 in pairs_of_cols:
            print(f'({col1}, {col2})')
            col1_vals = df[col1]
            col2_vals = df[col2]
            total_iterations = len(col1_vals)
            correlations = []
            cannot_calculate_due_to_invalid_length = 0
            for i in range(total_iterations):
                arr1 = CorrelationCalculator.get_non_nan_array(col1_vals[i])
                arr2 = CorrelationCalculator.get_non_nan_array(col2_vals[i])
                if len(arr1) == len(arr2):
                    correlations.append(pearsonr(arr1, arr2)[0])
                else:
                    cannot_calculate_due_to_invalid_length += 1
            if cannot_calculate_due_to_invalid_length == total_iterations:
                dict['correlations'].append({
                    'cols': f'({col1}, {col2})',
                    'col1': col1,
                    'col2': col2,
                    'number_unable_to_calculate': cannot_calculate_due_to_invalid_length,
                    'unable_to_calculate': True,
                    'mean_correlation': np.nan,
                    'std_dev_correlation': np.nan
                })
            else:
                temp_frame = pd.DataFrame(correlations)
                description = temp_frame.describe()
                dict['correlations'].append({
                    'cols': f'({col1}, {col2})',
                    'col1': col1,
                    'col2': col2,
                    'number_unable_to_calculate': cannot_calculate_due_to_invalid_length,
                    'mean_correlation': description[0]['mean'],
                    'std_dev_correlation': description[0]['std']
                })

        return dict

    @staticmethod
    def get_non_nan_array(arr):
        temp = np.array(arr)
        result = np.unique(temp[~np.isnan(temp)])
        return result
