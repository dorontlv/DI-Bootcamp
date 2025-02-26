'''

Exercises XP

What you will create
You will create a series of Python scripts that perform different statistical analyses, including data exploration, hypothesis testing, linear regression, ANOVA, and more.

'''


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


'''
Exercise 1: Basic Usage of SciPy

Task: Import the SciPy library and explore its version.

'''

import scipy

print(scipy.__version__)



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


'''

Exercise 2: Descriptive Statistics

Task: Given a sample dataset, calculate the mean, median, variance, and standard deviation using SciPy.

'''

import scipy.stats as stats
import numpy as np

data = [12, 15, 13, 12, 18, 20, 22, 21]

# using SciPy
mean_scipy = stats.tmean(data)
median_scipy = np.median(data)  # SciPy does not have a separate median function, so we use numpy
variance_scipy = stats.tvar(data)
std_dev_scipy = stats.tstd(data)



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


'''
Exercise 3: Understanding Distributions

Task: Generate a normal distribution using SciPy with a mean of 50 and a standard deviation of 10.
Plot this distribution.

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

table = np.random.normal(loc=50, scale=10, size=300)

# ploting
plt.hist(table, bins=40, color='blue')
plt.title('Histogram of a large sample with a normal distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


'''
Exercise 4: T-Test Application
Task: Perform a T-test on two sets of randomly generated data.

'''

# two sets of randomly generated data - but with a different mean value
data1 = np.random.normal(50, 10, 100)
data2 = np.random.normal(60, 10, 100)

from scipy.stats import ttest_ind

# Perform a T-test for independent groups
t_statistic, p_value = ttest_ind(data1, data2)

print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

alpha = 0.05  # significance level (5%)

if p_value < alpha:
    print("Reject the null hypothesis.")
else:
    print("Fail to reject the null hypothesis.")



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


'''
Exercise 5: Linear Regression Analysis
Objective: Apply linear regression to a dataset and interpret the results.

Task: Given a dataset of housing prices (house_prices) and their corresponding sizes (house_sizes), use linear regression to predict the price of a house given its size.

Questions:
What is the slope and intercept of the regression line?
Predict the price of a house that is 90 square meters.
Interpret the meaning of the slope in the context of housing prices.

'''

# Dataset:
house_sizes = [50, 70, 80, 100, 120]  # (in square meters)
house_prices = [150000, 200000, 210000, 250000, 280000]  # (in currency units)

# calculating the linear regression of the above datasets
slope, intercept, r_value, p_value, std_err = stats.linregress(house_sizes, house_prices)

print(f"The slope: {slope}")
print(f"The intercept: {intercept}")

x = 90  # a house that is 90 square meters
y = int(slope*x+intercept)
print(f"A {x} square meters house would cost: {y} dollars.")


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


'''
Exercise 6: Understanding ANOVA
Objective: Test understanding of ANOVA and its application.

Task:
Three different fertilizers are applied to three separate groups of plants to test their effectiveness.
The growth in centimeters is recorded.

Questions:
Perform an ANOVA test on the given data. What are the F-value and P-value?
Based on the P-value, do the fertilizers have significantly different effects on plant growth?
Explain what would happen if the P-value were greater than 0.05.

'''

# Dataset:
fertilizer_1 = [5, 6, 7, 6, 5]
fertilizer_2 = [7, 8, 7, 9, 8]
fertilizer_3 = [4, 5, 4, 3, 4]

# ANOVA test
f_value, p_value = stats.f_oneway(fertilizer_1, fertilizer_2, fertilizer_3)

print("F-value:", f_value)
print("P-value:", p_value)

'''

Explanation taken from the Octopus website:

A high F-value (as seen in the example, 28.52) suggests that there is a significant variation among the group means, which indicates that at least one group mean is different from the others.

P-value (2.75e-05): This value indicates the probability of observing such an F-value assuming the null hypothesis is true. The null hypothesis for ANOVA states that all group means are equal. A very small p-value (in this case, 5.40e-07) means that the probability of observing such data if the null hypothesis were true is extremely low. Consequently, we reject the null hypothesis and conclude that there are significant differences among the group means.

If P-value were greater than 0.05, it would mean that the probability of observing such data if the null hypothesis were true is rather high, so we would NOT reject the null hypothesis that all group means are equal.

''' 



