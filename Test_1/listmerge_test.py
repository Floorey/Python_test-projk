import numpy as np




class AnalyseValues():
    def __init__(self):
        self.max_value1 = None
        self.max_value2 = None


    @staticmethod
    def analyse_values_max(array_1, array_2):
        max_value1 = np.max(array_1)
        max_value2 = np.max(array_2)
        return max_value1, max_value2
    


   
   

array_1 = np.random.rand(10)
array_2 = np.random.rand(50)
print(f"Generated values: {array_1}")
print(f"Generated values: {array_2}")




analyse_instance = AnalyseValues()
result_1, result_2 = AnalyseValues.analyse_values_max(array_1, array_2)

analyse_instance = result_1
analyse_instance = result_2

print(f"Max value 1 is: {result_1}")
print(f"Max value 2 is: {result_2}")




