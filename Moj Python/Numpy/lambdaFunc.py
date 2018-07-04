# Program to show the use of lamba functions
# lambda argumants: expression

double = lambda x: x * 2 # x-argument, x*2-expression

# output : 10
print(double(5))

# nearly the same as
def double(x):
    return x * 2

# lambda use  with filter()
"""
The filter() function in Python takes in a function and a list as arguments.
The function is called with all the items in the list and 
a new list is returned which contains items for which the function 
evaluats to True
"""
# program to filter out only the even items from a list
my_list = [1, 5, 4, 6, 8, 11, 3, 12]

new_list = list(filter(lambda x: (x%2 == 0), my_list))

print(new_list)

'''
lambda use with map()
The function is called with all the items in the list and 
a new list is returned which contains items returned 
by that function for each item.
'''
# program to double each item in a list using map()
map_list = [1, 5, 4, 6, 8, 11, 3, 12]

new_map_list = list(map(lambda x: x * 2, map_list))

print(new_map_list)