import random
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from abc import ABC, abstractmethod


def creat_values():
    return [random.uniform(0, 1) for _ in range(10)]


class DataInterface(ABC):
    @abstractmethod
    def save(self, data, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass


class SaveValues(DataInterface):
    def save(self, data, filename, append=True):
        mode = 'a' if append else 'w'
        with open(filename, mode) as file:
            for key, values in data.items():
                line = f'{key}: {",".join(map(str, values))}\n'
                file.write(line)

    def load(self, filename):
        return {}

class LoadValues(DataInterface):
    def load(self, filename): 
        result = {}
        with open(filename, 'r') as file:
            for line in file:
                key, values_str = line.strip(':')
                values = list(map(int, values_str.split(',')))
                result[key] = values
            return result 

    def save(self, data, filename):
        pass  

class AnalyseValues():
    def __init__(self):
        pass
    @classmethod
    def analyse_max_value(cls, values):
        return np.max(values)
    @classmethod 
    def analyse_lowest_value(cls, values):
        return np.min(values)
    @classmethod    
    def analyse_var(cls, values):
        return np.var(values)
    @classmethod
    def analyse_std_dev(cls, values):
        return np.std(values)
    
        
        
class Main():
    def __init__(self, data_saver):
        self.values1 = creat_values()
        self.values2 = creat_values()
        self.values3 = creat_values()
        self.analyse_instance = AnalyseValues()
        self.saver = data_saver


    def perform_analysis(self, values):
        max_value = self.analyse_instance.analyse_max_value(values)
        min_value = self.analyse_instance.analyse_lowest_value(values)
        var_value = self.analyse_instance.analyse_var(values)
        std_value = self.analyse_instance.analyse_std_dev(values)
        return max_value, min_value, var_value, std_value
    
    def run(self, filename, append=True):
        result1 = self.perform_analysis(self.values1)
        result2 = self.perform_analysis(self.values2)
        result3 = self.perform_analysis(self.values3)

        keys = ['values1', 'values2', 'values3']
        values_dict = {'values1': result1, 'values2': result2, 'values3': result3}

        self.saver.save(values_dict, filename, append=append)

        loaded_data = self.saver.load(filename)
        pprint(loaded_data)

   

        


if __name__ == "__main__":
    saver_instance = SaveValues()
    main_instance = Main(saver_instance)
    main_instance.run('example.txt', append=True)
