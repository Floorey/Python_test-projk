import random
import pickle
from datetime import datetime
import statistics
from collections import Counter


class DataAnalysis:
    def __init__(self, values):
        self.values = values

    def analyse_low_values(self):
        for index, value in enumerate(self.values):
            if value < 295:
                print(f'Index {index}: {value} is too low!')

    def analyse_high_values(self):
        for index, value in enumerate(self.values):
            if value > 495:
                print(f'Index {index}: {value} is too high!')

    def analyse_most_value(self):
        value_counts = Counter(self.values)
        most = max(value_counts, key=value_counts.get)
        print(f'The most frequent value: {most} ')

    def analyse_less_value(self):
        value_counts = Counter(self.values)
        less = min(value_counts, key=value_counts.get)
        print(f'The less frequent value is: {less} ')

    def analyse_max_value(self):
        max_value = max(self.values)
        print(f'The highest value is: {max_value}')

    def analyse_low_value(self):
        min_value = min(self.values)
        print(f'The lowest value is: {min_value}')

    def analyse_average_value(self):
        average_value = sum(self.values) / len(self.values)
        print(f'The average value is: {average_value}')

    def analyse_variance(self):
        variance_value = statistics.variance(self.values)
        print(f'The variance is: {variance_value}')

    def analyse_2sd(self):
        mean_value = statistics.mean(self.values)
        std_deviation = statistics.stdev(self.values)

        lower_bound = mean_value - (2 * std_deviation)
        upper_bound = mean_value + (2 * std_deviation)

        print(f'The 2SD area is: ({lower_bound}, {upper_bound})')
    
    def analyse_median(self):
        median_value = statistics.median(self.values)
        print(f'The median value is: {median_value}')
    
    def analyse_std_dev(self):
        std_deviation = statistics.stdev(self.values)
        print(f'The std_deviation is: {std_deviation}')




def create_values():
    return [random.randint(1, 500) for _ in range(50)]

def merge_lists(arrayA, arrayB):
    return sorted(set(arrayA + arrayB))

def save_to_file(data, folder_path):
    current_date_and_time = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    filename = f'list_file_{current_date_and_time}.pkl'

    with open(folder_path + filename, 'wb') as my_file:
        pickle.dump(data, my_file)

    return filename

def load_from_file(file_path):
    with open(file_path, 'rb') as my_file:
        return pickle.load(my_file)

def main():
    values1 = create_values()
    values2 = create_values()
    merged_values = merge_lists(values1, values2)

    folder_path = "C:/Users/lukas/OneDrive/Dokumente/Python_test-projk/Test_1"
    filename = save_to_file(merged_values, folder_path)
    loaded_lists = load_from_file(folder_path + filename)

    data_analysis = DataAnalysis(loaded_lists)
    data_analysis.analyse_low_values()
    data_analysis.analyse_high_values()
    data_analysis.analyse_most_value()
    data_analysis.analyse_less_value()
    data_analysis.analyse_max_value()
    data_analysis.analyse_low_value()
    data_analysis.analyse_average_value()
    data_analysis.analyse_2sd()
    data_analysis.analyse_variance()
    data_analysis.analyse_median()
    data_analysis.analyse_std_dev()
  

    print(f'Loaded list of values: {loaded_lists}')

if __name__ == "__main__":
    main()