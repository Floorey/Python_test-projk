from timeit import timeit

my_list = list(range(100_000))
my_set = set(range(100_000))



list_time = timeit('99_999 in my_list', number=1000, globals=globals())
print (f'List: {list_time: .6} seconds')

set_time = timeit('99_999 in my_set', number=1000, globals=globals())
print (f'Set: {set_time: .6f} seconds')


speed_difference = (list_time-set_time)/list_time*100

print(f'{speed_difference: .3f} % faster')
print(type(my_set))