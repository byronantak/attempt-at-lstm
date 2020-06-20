import pandas as pd


class StatsCalculator:
    @staticmethod
    def get_descriptive_statistics_for_data(data):
        running_mean = 0
        count = 0
        min = 999999999
        min_row_nan = 999999999
        max = -1 * min
        max_row_nan = -1 * min
        running_std_dev = 0
        nan_total = 0
        for temp_arr in data:
            temp = pd.DataFrame(temp_arr)
            description = temp.describe()
            nan_count = temp.isnull().sum()[0]
            nan_total += nan_count
            if nan_count > max_row_nan:
                max_row_nan = nan_count
            if nan_count < min_row_nan:
                min_row_nan = nan_count
            running_mean += description[0]['mean']
            running_std_dev += description[0]['std']
            if min > description[0]['min']:
                min = description[0]['min']
            if max < description[0]['max']:
                max = description[0]['max']
            count += 1
        std_dev_mean = running_std_dev / count
        nan_count_avg = nan_total / count
        mean = running_mean / count
        print(f'min: {min}')
        print(f'max: {max}')
        print(f'mean: {mean}')
        print(f'std_dev_mean: {std_dev_mean}')
        print(f'nan_count_avg: {nan_count_avg}')
        return [mean, std_dev_mean, min, max, nan_count_avg, min_row_nan, max_row_nan]
