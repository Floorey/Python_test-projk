import random
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


def creat_values():
    return [random.uniform(0, 1) for _ in range(10)]




class SaveValues():
    def __init__(self, filename):
        self.filename = filename

    def save_lists(self, keys, values_lists):
        with open(self.filename, 'w') as file:
            for key, values in zip(keys, values_lists):
                line = f'{key}: {",".join(map(str, values))}\n'
                file.write(line)


class LoadValues():
    def __init__(self, filename):
        self.filename = filename

    def load_lists(self):
        result = {}
        with open(self.filename, 'r') as file:
            for line in file:
                key, values_str = line.strip(':')
                values = list(map(int, values_str.split(',')))
                result[key.strip()] = values
            return result

class AnalyseValues():
    def __init__(self):
        pass

    def analyse_max_value(self, values):
        return np.max(values)
     
    def analyse_lowest_value(self, values):
        return np.min(values)
        
    def analyse_var(self, values):
        return np.var(values)
    
    def analyse_std_dev(self, values):
        return np.std(values)
    
        
        
class Main():
    def __init__(self):
        self.values1 = creat_values()
        self.values2 = creat_values()
        self.values3 = creat_values()
        self.analyse_instance = AnalyseValues()
        self.saver = SaveValues('examle.txt')

    def perform_analysis(self, values):
        max_value = self.analyse_instance.analyse_max_value(values)
        min_value = self.analyse_instance.analyse_lowest_value(values)
        var_value = self.analyse_instance.analyse_var(values)
        std_value = self.analyse_instance.analyse_std_dev(values)
        return max_value, min_value, var_value, std_value
    
    def run(self):
        result1 = self.perform_analysis(self.values1)
        result2 = self.perform_analysis(self.values2)
        result3 = self.perform_analysis(self.values3)

        keys = ['values1', 'values2', 'values3']
        values_lists = [result1, result2, result3]
        self.saver.save_lists(keys, values_lists)

        return result1, result2, result3
        




if __name__ == "__main__":
    main_instance = Main()
    results = main_instance.run()
    pprint(results)
