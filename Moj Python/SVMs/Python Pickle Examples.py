import pickle
import numpy as np

# pickle list object

numbers_list = np.arange(1,6)
print(numbers_list)
list_pickle_path = 'list_pickle.pkl'

# Create an variable to pickle and open it in write mode
# list_pickle = open(list_pickle_path, 'wb')
# pickle.dump(numbers_list, list_pickle)
# list_pickle.close()

# unpickling the list object

# Need to open the pickled list object into read mode

list_unpickle = open(list_pickle_path, 'rb')

# load the unpickle object into a variable
numbers_list = pickle.load(list_unpickle)

print("Numbers List :: ", numbers_list)

