import random
import pickle
from datetime import datetime


# value creation
def creat_values():
    values = [random.randint(1,100) for i in range (50)]
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


print(loaded_lists)

def analyse():
    for index in loaded_lists:
        if index < 10:
            print('value too low!')
        else:
            print(loaded_lists[10])
            break


analyse()

       
            

 

