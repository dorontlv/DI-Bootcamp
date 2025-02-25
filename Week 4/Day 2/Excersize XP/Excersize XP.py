'''

Exercises XP

Understanding and calculating the determinant and inverse of matrices.
Application of these concepts in linear algebra and their significance.

'''

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
Exercise 1 : Matrix Operations


In this exercise, you’ll work with a 3x3 matrix.
Here’s a brief explanation of the concepts:

Determinant: The determinant is a value that can be computed from the elements of a square matrix.
It provides important information about the matrix, such as whether it has an inverse, and is used in various areas like linear algebra and calculus.
To understand more about it you can watch this video.
Inverse of a Matrix: The inverse of a matrix is a matrix that, when multiplied with the original matrix, results in an identity matrix.
Not all matrices have inverses.
The inverse is crucial in solving systems of linear equations.
Create a 3x3 matrix and perform the following operations:

Calculate the determinant.
Find the inverse of the matrix.

'''
import numpy as np

mat = np.array([[1, 2, 3],[0, 1, 4],[5, 6, 0]])  # a 3X3 matrix

# Calculate the determinant of a matrix
determinant = np.linalg.det(mat)


# Calculate the inverse of a matrix
inverse_matrix = np.linalg.inv(mat)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


'''

Exercise 2 : Statistical Analysis

In this exercise, you’ll calculate statistical measures for a dataset:

Mean: The average value of a dataset.
Median: The middle value in a dataset when it is arranged in ascending or descending order.
Standard Deviation: A measure of the amount of variation or dispersion in a set of values.

Using NumPy, generate an array of 50 random numbers and compute:
The mean and median.
The standard deviation.

'''


arr = np.random.randint(1,1000,50)

result = np.mean(arr)
result = np.median(arr)
result = np.std(arr)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


'''

Exercise 3 : Date Manipulation

Create a NumPy array of dates for the month of January 2023.
Convert these dates to another format (e.g., YYYY/MM/DD).

'''

import pandas as pd
import datetime

# creating date ranges
dates = np.arange('2023-01', '2023-02', dtype='datetime64[D]')
print("January 2020 Dates:\n", dates)
dates

# Converting dates to a different format
formatted_dates = np.datetime_as_string(dates, unit='D')

# Reformat the strings to 'YYYY/MM/DD'
formatted_dates = np.char.replace(formatted_dates, '-', '/')



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


'''
Exercise 4 : Data Manipulation with NumPy and Pandas

Create a DataFrame with random numbers and perform:

Conditional selection of data.
Aggregation functions like sum and average.

'''

import pandas as pd

# 15 random numbers
data = pd.DataFrame({'ordinal': np.arange(1,15+1), 'randnum': np.random.randint(1,100,15)})

# conditionaly choose where randnum is more than 30, and calculate the sum of the randnum
result = data[data['randnum']>30]['randnum'].sum()

print(result)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


'''
Exercise 5 : Image Representation

Explain how images are represented in NumPy arrays and demonstrate with a simple example (e.g., creating a 5x5 grayscale image).

'''

# an image is a 2D array of pixels, where each pixel has a color (in our example, it's a gray color).
# the values are between 0 to 255.

import numpy as np
import matplotlib.pyplot as plt

# create a 5x5 array with random values between 0 and 255 (it represents the gray colors)
grayscale_image = np.random.randint(0, 255+1, size=(5, 5))

# display the image using matplotlib
plt.imshow(grayscale_image, cmap='gray')
plt.show()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


'''

Exercise 6 : Basic Hypothesis Testing

Create a sample dataset to test the effectiveness of a new training program on employee productivity:


import numpy as np

# Productivity scores of employees before the training program
productivity_before = np.random.normal(loc=50, scale=10, size=30)

# Productivity scores of the same employees after the training program
productivity_after = productivity_before + np.random.normal(loc=5, scale=3, size=30)

# Your task is to formulate a hypothesis regarding the training program's effectiveness 
# and test it using basic statistical functions in NumPy.


Given a dataset, formulate a simple hypothesis and test it using basic statistical functions in NumPy.

'''

import numpy as np
from scipy import stats

# Productivity scores of employees before the training program
productivity_before = np.random.normal(loc=50, scale=10, size=30)

# Productivity scores of the same employees after the training program
productivity_after = productivity_before + np.random.normal(loc=5, scale=3, size=30)

# Calculate the mean difference
mean_difference = np.mean(productivity_after - productivity_before)

# Perform a paired t-test
t_statistic, p_value = stats.ttest_rel(productivity_after, productivity_before)

print(f"Mean difference: {mean_difference}")
print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


'''
Exercise 7 : Complex Array Comparison

Create two arrays and perform element-wise comparison to find which elements are greater in the first array.

The expected output is a boolean array showing which elements in the first array are greater than the corresponding elements in the second array.

'''

# matrices - shape 3X4 - with random numbers
mat1 = np.random.randint(1,100,12).reshape(3,4)
mat2 = np.random.randint(1,100,12).reshape(3,4)

print(mat1 > mat2)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


'''
Exercise 8 : Time Series Data Manipulation

Generate time series data for the year 2023.

Demonstrate slicing for the following intervals:
January to March
April to June
July to September
October to December

Generate a time series data for a specific period and demonstrate how to slice this data for different intervals.

'''
# the range of dates
dates = np.arange('2023-01', '2024-01', dtype='datetime64[D]')

# convert into strings
string_dates = np.datetime_as_string(dates, unit='D')

# slicing the data
string_dates[('2023-01-01'<=string_dates) & (string_dates<='2023-03-31')]
string_dates[('2023-04-01'<=string_dates) & (string_dates<='2023-06-30')]
string_dates[('2023-07-01'<=string_dates) & (string_dates<='2023-09-30')]
string_dates[('2023-10-01'<=string_dates) & (string_dates<='2023-12-31')]



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
Exercise 9 : Data Conversion

Demonstrate how to convert a NumPy array to a Pandas DataFrame and vice versa.

'''


arr1 = np.array([[1,2,3,4],[5,6,7,8]])
df = pd.DataFrame(arr1)  # convert an array to a DataFrame 
arr2 = np.array(df)  # convert a DataFrame to an array


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
Exercise 10 : Basic Visualization

Use Matplotlib to visualize a simple dataset created with NumPy (e.g., a line graph of random numbers).

'''

# random numbers
arr = np.random.randint(1,100,10)

import matplotlib.pyplot as plt

# plot them onto a line graph
plt.plot(arr, marker='o', linestyle='-')
plt.title('Simple line graph')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()


