'''
First steps with Python
Search Heuristics - University of the Basque Country (UPV/EHU)
Inigo Lopez-Gazpio
'''


#---------------------------------------------------------------------------
# This script aims to present the general sintax of Python briefly.
# It condenses common operations needed during the Subject
#---------------------------------------------------------------------------


'''
About Python

Python is an interpreted high-level general-purpose programming language.
Its design philosophy emphasizes code readability with its notable use of significant indentation.
Based on an object-oriented approach it aims to help programmers write clear, logical code for small and large-scale projects

Code can be executed line-by-line or segment-by-segment using ipython interpreter, copying code in an IDE and running %paste command.
Code can be executed all-in-one through ipython interpreter running the %run FILE command or through python interpreter in a terminal.
'''


'''
Working directory and importing extra files

Caution must be taken to determine the working directory from which python files are executed.
In all cases the working directory is set to the placement of the terminal from which python commands are launched.
Extra files imported within scripts must be located taking the working directory as root placement.

In this regard, shell terminal commands can be launched using the % symbol in the ipython interpreter, such as: %ls, %pwd, %cd Path, etc
'''


'''
Variables

In Python, there is no need to allocate memory for variables, or "worry" about their types.
Variable types can be checked with type(variable) function or isinstance(variable,Type)
'''
var_1 = 2
var_2 = "two"
var_1 = "Three"
var_2 = 3.4

type(var_1)
type(var_2)

isinstance(var_1, int)
isinstance(var_2, float)
isinstance(var_2, str)

'''
Python also incorporates the chance to use collections, such as lists, typles or dictionaries.
To declare lists, just use the [ ] syntax, and then call append method to add elements.
More complex collections can be found in the collections library, check https://docs.python.org/3/library/collections.html
'''

mylist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
mylist = [True, True, True, True]
mylist.append(False)


# There are multiple commands to work with lists
list_1 = range(1,10)               # Numbers from 1 to 10
list_2 = range(1, 10, 2)           # Sequence from 1 to 10 adding 2 in each step

# we can produce lists of random numbers, better to use random library for this
import random
list_3 = [random.uniform(1,10) for _ in range(10) ]          # Generate 10 random numbers between 1 and 10
list_4 = list_3 * 2                                          # Repeat X times the content of a vector
list_5 = [random.choice(range(1,10)) for _ in range(10) ]    # Generate 10 random discrete integers between 1 and 10


# for heavy mathematical operations it is better to use specialized libraries such as numpy
import numpy as np
array = np.random.random(10)
matrix = np.random.random((10,10))

# We access the components with brackets
array[2]
matrix[1,3]

# For matrices, we can access an entire row / column
matrix[1,:]
matrix[:,1]

# And to do operations over multiple elements
matrix[1,:] = 10 # Set to 10 all the elements of row 1.
matrix[:,2] += 10 # Add 10 to all the elements of column 2.

# Square all the elements
array = array ** 2

# FOR structures can be used

for index, value in enumerate(array):
  array[index] = value ** 2

# Another option, elegant, efficient, and necessary in some cases is to use functions of the family map / apply
np.apply_along_axis(np.mean, 1, matrix) # compute mean over columns
np.apply_along_axis(np.mean, 0, matrix) # compute mean over rows

# Definition of a function
def square(x):
    return x**2

square(2)
square(np.array([0,1,2,3,4,5,6,7,8,9]))
matrix[:,1] = np.apply_along_axis(square, 0, matrix[:,1])


# Making validations or conditional changes on a vector (or matrix)
list_6 = np.array(range(1,10))  # Create a vector of numbers between 1 and 10 and order randomly.
list_6 < 5                      # Check if the elements are lower than 5.
np.where(list_6 < 5)     # Get the indexes with value lower than 5.
list_6[list_6 < 5] = 0  # Set to 0 the element lower than 5.

# Get the index of all the elements with value X
matrix = np.array([ [11, 12, 13, 14, 15, 16, 17], [15, 11, 12, 14, 15, 16,17]])
matrix.shape
matrix.size
indexes = np.where(matrix == 15)



'''
Python is also a very robust language in terms of OOP, supporting multiple inheritance and polymorphism
'''
from abc import ABC
class Person(ABC):
    def __init__(self, name, age, cities, data):
        self.name = name
        self.age = age
        self.cities = cities
        self.data = data

person1 = Person("Mike", 29, ["Donostia","Bilbao"], np.array([[0,1,2,3],[4,5,6,7]]))
person2 = Person("Mary", 30, ["Donostia","Bilbao","Madrid"], np.zeros((3,3)))

people = list()
people.append(person1)
people.append(person2)
del people[1]
print(len(people))
print(people[0].name)


'''
Python also provides an extensive API to work with linear and non-linear collections, check https://docs.python.org/3/tutorial/datastructures.html
'''