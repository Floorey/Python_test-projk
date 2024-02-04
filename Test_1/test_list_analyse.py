import random
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint


def creat_values():
    return [random.uniform(0, 1) for _ in range(10)]


values1 = creat_values()
values2 = creat_values()
values3 = creat_values()


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
        self.max_value = None
        self.min_value = None
        self.var_value = None
        self.std_value = None

    def analyse_max_value(self, values):
        return np.max(values)
     
    def analyse_lowest_value(self, values):
        return np.min(values)
        
    def analyse_var(self, values):
        return np.var(values)
    
    def analyse_std_dev(self, values):
        return np.std(values)
        
class main():
    

    analyse_instance = AnalyseValues()
    result_1 = analyse_instance.analyse_max_value(values1)
    result_2 = analyse_instance.analyse_max_value(values2)
    result_3 = analyse_instance.analyse_max_value(values3)

    analyse_instance2 = AnalyseValues()
    result_1_low = analyse_instance2.analyse_lowest_value(values1)
    result_2_low = analyse_instance2.analyse_lowest_value(values2)
    result_3_low = analyse_instance2.analyse_lowest_value(values3)

    analyse_instance3 = AnalyseValues()
    result_1_var = analyse_instance3.analyse_var(values1)
    result_2_var = analyse_instance3.analyse_var(values2)
    result_3_var = analyse_instance3.analyse_var(values3)

    analyse_instance4 = AnalyseValues()
    result_1_std = analyse_instance4.analyse_std_dev(values1)
    result_2_std = analyse_instance4.analyse_std_dev(values2)
    result_3_std = analyse_instance4.analyse_std_dev(values3)

    saver = SaveValues('example.txt')
    keys = ['values1', 'values2', 'values3']
    values_lists = [values1, values2, values3]
    saver.save_lists(keys, values_lists)
    


    print(f'The max value1 is: {result_1}')
    print(f'The max value2 is: {result_2}')
    print(f'The max value3 is: {result_3}')  
    print(f'The lowest value1 is: {result_1_low}')  
    print(f'The lowest value2 is: {result_2_low}')
    print(f'The lowset value3 is : {result_3_low}')
    print(f'The varianz of value1 is: {result_1_var}')
    print(f'The varianz of value2 is: {result_2_var}')
    print(f'The varianz of value3 is: {result_3_var}')
    print(f'The Std-Deviation of value1 is: {result_1_std}')
    print(f'The Std-Deviation of value2 is: {result_2_std}')
    print(f'The Std-Deviation of value3 is: {result_3_std}')
    pprint(values1)
    pprint(values2)
    pprint(values3)

    plt.scatter(range(len(values1)), values1,
                color='green', marker='.', s=10)
    
    plt.scatter(range(len(values2)), values2,
                color='red', marker='.', s=12)
    
    plt.scatter(range(len(values3)), values3,
                color='blue', marker='.', s=12)


    plt.title('destribution of values')
    plt.xlabel('Index')
    plt.ylabel('Value') 
    plt.show()

    


if __name__ == "__main__":
    main()