import numpy as np
import random






array_1 = np.random.rand(50)
array_2 = np.random.rand(50)
print(f"Generated values: {array_1}")
print(f"Generated values: {array_2}")


class AnalyseValues():
    def __init__(self):
        self.max_value1 = None
        self.max_value2 = None
        self.var_value1 = None
        self.var_value2 = None
        self.std_dev_1 = None
        self.std_dev_2 = None

    def analyse_values_max(array_1, array_2):
        max_value1 = np.max(array_1)
        max_value2 = np.max(array_2)
        return max_value1, max_value2
      
    def analyse_variance(array_1, array_2):
        var_value1 = np.var(array_1)
        var_value2 = np.var(array_2)
        return var_value1, var_value2
    
    def std_deviation(array1, array2):
        std_dev_1 = np.std(array_1)
        std_dev_2 = np.std(array_2)
        return std_dev_1, std_dev_2
   
   
analyse_instance = AnalyseValues()
analyse_instance2 = AnalyseValues()
analyse_instance3 = AnalyseValues()
result_1, result_2 = AnalyseValues.analyse_values_max(array_1, array_2)
result_var_1, result_var_2 = AnalyseValues.analyse_variance(array_1, array_2)
result_std_1, result_std_2 = AnalyseValues.std_deviation(array_1, array_2)

analyse_instance = result_1
analyse_instance = result_2
analyse_instance2 = result_var_1
analyse_instance2 = result_var_2
analyse_instance3 = result_std_1
analyse_instance3 = result_std_2


print(f"Max value 1 is: {result_1}")
print(f"Max value 2 is: {result_2}")
print(f'The varianz for array1 is: {result_var_1}')
print(f'The varianz for array2 is: {result_var_2}')
print(f'The standard-deviation of array1 is: {result_std_1}')
print(f'The standrad-deviation of array2 is: {result_std_2}')




