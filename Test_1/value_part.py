import numpy as np
import pandas as pd
from abc import ABC, abstractclassmethod

class DataInterface(ABC):
    @abstractclassmethod
    def save(self, data, filename):
        pass

    @abstractclassmethod
    def load(self, filename):
        pass

class SaveValues(DataInterface):
    def save(self, data, filename):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    def load(self, filename):
        return pd.read_csv(filename)


class Main():
    def __init__(self, data_saver):
        self.saver = data_saver

    def perform_analysis(self, values):
        max_value = np.max(values)
        min_value = np.min(values)
        var_value = np.var(values)
        std_value = np.std(values)
        sec_std_value = np.std(values) * 2
        mean_value = np.mean(values)
        return max_value, min_value, var_value, std_value, sec_std_value, mean_value

    def run(self, filename):
        # Daten generieren
        values1 = np.random.uniform(0, 1, 50)
        values2 = np.random.uniform(0, 1, 50)
        values3 = np.random.uniform(0, 1, 50)

        # Daten analysieren
        result1 = self.perform_analysis(values1)
        result2 = self.perform_analysis(values2)
        result3 = self.perform_analysis(values3)

        # Formatieren und speichern der Daten im CSV-Format
        keys = ['values1', 'values2', 'values3']
        values_dict = {'values1': result1, 'values2': result2, 'values3': result3}
        formatted_data = {key: [key] + list(values) for key, values in values_dict.items()}
        self.saver.save(formatted_data, filename)

        print("Daten wurden gespeichert.")

if __name__ == "__main__":
    saver_instance = SaveValues()
    main_instance = Main(saver_instance)
    main_instance.run('exaplle.csv')