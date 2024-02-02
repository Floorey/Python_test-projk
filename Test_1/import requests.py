import random
import pickle
from datetime import datetime


# value creation
def creat_values():
    values = [random.uniform(1,100) for i in range (50)]
    return values

#creat varible lists
values1 = creat_values()
values2 = creat_values()

#merge of the two lists
def merge_lists(arrayA, arrayB):
    return sorted(set(arrayA + arrayB))

values3 = merge_lists(values1,values2)

#create a file with date and time stamp
current_date_and_time = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
filename = f'list_file_{current_date_and_time}.pkl'

#save the list
folder_path = "C:/Users/lukas/OneDrive/Dokumente/Python_test-projk/Test_1"

with open (folder_path + filename, 'wb') as my_file:
       pickle.dump(values3, my_file)


with open (folder_path + filename, 'rb') as my_file:
       loaded_lists = pickle.load(my_file)



def analyse_low_values():
    for index, value in enumerate(loaded_lists):
        if value < 10:
            print(f'Index {index}: {value} is too low!')

def analyse_high_values():
    for index, value in enumerate(loaded_lists):
        if value > 90:
            print(f'Index {index}: {value} is too high!')


def analyse_most_value():
    most = max(set(loaded_lists), key=loaded_lists.count)
    print(f'The most frequent value: {most} ')
     
def analyse_max_value():
    max_value = max(loaded_lists)
    print(f'The highest value is: {max_value}')   

def analyse_low_value():
    min_value = min(loaded_lists)
    print(f'The lowest value is: {min_value}')      
          

def main():
    analyse_low_values()
    analyse_high_values()
    analyse_most_value()
    analyse_max_value()
    analyse_low_value()

if __name__ == "__main__":
    main()

       
            

 



       
            

 

