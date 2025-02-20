'''

Exercises XP


What you will learn
Identify and remove duplicate entries in the Titanic dataset.

For all of the below exercises, you will use the Titanic dataset (train.csv), so load it beforehand on your notebook.

'''

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


'''

Exercise 1: Duplicate Detection and Removal

Objective: Identify and remove duplicate entries in the Titanic dataset.

'''

import pandas as pd

# Load the Titanic dataset
titanic_data = pd.read_csv('train.csv')

number_of_rows = len(titanic_data)

# Identify if there are any duplicate rows based on all columns.
number_of_duplicates = titanic_data.duplicated().sum()

# Remove duplicate rows
if number_of_duplicates:
    titanic_data = titanic_data.drop_duplicates()

number_of_rows_after_dropping = len(titanic_data)

if number_of_rows != number_of_rows_after_dropping:
    print("We removed the duplicates.")




print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


'''

Exercise 2: Handling Missing Values

Identify columns in the Titanic dataset with missing values.
Explore different strategies for handling missing data, such as removal, imputation, and filling with a constant value.

'''

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer  # Import SimpleImputer

# Load the Titanic dataset
titanic_data = pd.read_csv('train.csv')

# are there NA values ?
missing_data = titanic_data.isnull()

# get the sum of NA on every column
missing_counts = missing_data.sum()

# ~~~~~~~~~~~~

# We can do this:
# Remove rows with missing values
df_cleaned = titanic_data.dropna()

# OR:
# we can do this:
titanic_data_only_numbers = titanic_data.select_dtypes(include=np.number)  # select columns with numbers only

df_filled = titanic_data_only_numbers.fillna(titanic_data_only_numbers.mean())  # fill them with the mean


# ~~~~~~~~~~~~

# OR:
# use SimpleImputer 

titanic_data_numerical = titanic_data.select_dtypes(include=['number'])    # select columns with numbers only

imputer = SimpleImputer(strategy='mean') # Create an imputer instance
titanic_data_numerical = imputer.fit_transform(titanic_data_numerical) # Impute missing values



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


'''
Exercise 3: Feature Engineering

Create new features, such as Family Size from SibSp and Parch, and Title extracted from the Name column.
Convert categorical variables into numerical form using techniques like one-hot encoding or label encoding.
Normalize or standardize numerical features if required.

'''


import pandas as pd
# import numpy as np


# Load the Titanic dataset
titanic_data = pd.read_csv('train.csv')

# Columns:
# SibSp
# Parch
# Calculate the family size:
titanic_data['Family Size'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1  # including the passanger itself

# ~~~~~~~~~~~~~~~~~

# Convert categorical variables into numerical form using techniques like one-hot encoding or label encoding.

# This will replace the 'Embarked' column with the following columns: Embarked_C  Embarked_Q  Embarked_S
titanic_data = pd.get_dummies(titanic_data, columns=['Embarked'])

# There's also the "label encoding" technique.
# It just gives a different numerical label to each category.
# I implemented it on excersie 6

print(titanic_data)

# ~~~~~~~~~~~~~~~~~

# Normalize or standardize numerical features if required.

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Normalizing the 'Age' column
titanic_data['Age_normalized'] = scaler.fit_transform(titanic_data[['Age']])

print(titanic_data)



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


'''

Exercise 4: Outlier Detection and Handling

Use statistical methods to detect outliers in columns like Fare and Age.
Decide on a strategy to handle the identified outliers, such as capping, transformation, or removal.
Implement the chosen strategy and assess its impact on the dataset.

'''


import pandas as pd


# Load the Titanic dataset
titanic_data = pd.read_csv('train.csv')


# Let's identify and handle outliers in 'Fare' column

# First quartile (Q1) and the third quartile (Q3) for the 'Fare' column
Q1 = titanic_data['Fare'].quantile(0.25)
Q3 = titanic_data['Fare'].quantile(0.75)
# Compute the interquartile range (IQR) for the Fare column.
IQR = Q3 - Q1
# Define the lower bound and upper bound for potential outliers based on the IQR.
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# removing outliers
titanic_data = titanic_data[(titanic_data['Fare'] >= lower_bound) & (titanic_data['Fare'] <= upper_bound)]

print(titanic_data)

# We can do the same on the "Age" column.



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''

Exercise 5: Data Standardization and Normalization

Assess the scale and distribution of numerical columns in the dataset.
Apply standardization to features with a wide range of values.
Normalize data that requires a bounded range, like [0, 1].

'''


import pandas as pd
# import numpy as np

# Load the Titanic dataset
titanic_data = pd.read_csv('train.csv')


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

scaler = MinMaxScaler()

# we can also use StandardScaler:
# scaler = StandardScaler()

# adding a column - where the age is normalized
titanic_data['Age_normalized'] = scaler.fit_transform(titanic_data[['Age']])

print(titanic_data)



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''
Exercise 6: Feature Encoding

Identify categorical columns in the Titanic dataset, such as Sex and Embarked.
Use one-hot encoding for nominal variables and label encoding for ordinal variables.
Integrate the encoded features back into the main dataset.

'''


import pandas as pd

# Load the Titanic dataset
titanic_data = pd.read_csv('train.csv')


# ~~~~~~~~~~

# Use one-hot encoding for nominal variables

# This will replace the 'Parch' column with the following columns: Parch_0  Parch_1  Parch_2  Parch_3  Parch_4  Parch_5  Parch_6
titanic_data = pd.get_dummies(titanic_data, columns=['Parch'])

print(titanic_data)

# ~~~~~~~~~~

# use label encoding for ordinal variables.
    
from sklearn.preprocessing import LabelEncoder

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Fit the LabelEncoder on the 'Embarked' column and transform it
titanic_data['Embarked_Label'] = label_encoder.fit_transform(titanic_data['Embarked'])

# Fit the LabelEncoder on the 'Sex' column and transform it
titanic_data['Sex_Label'] = label_encoder.fit_transform(titanic_data['Sex'])

print(titanic_data)



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


'''
Exercise 7: Data Transformation for Age Feature

Create age groups (bins) from the Age column to categorize passengers into different age categories.
Apply one-hot encoding to the age groups to convert them into binary features.
    
'''

import pandas as pd

# Load the Titanic dataset
titanic_data = pd.read_csv('train.csv')


# Define bin edges
bins = [0, 18, 35, 50, 65, 100]

# Define labels for the bins
labels = ['Child', 'Young Adult', 'Adult', 'Middle Aged', 'Senior']

# creating the Age Group column
titanic_data['Age Group'] = pd.cut(titanic_data['Age'], bins=bins, labels=labels, right=False)

# Apply one-hot encoding to the age groups
titanic_data = pd.get_dummies(titanic_data, columns=['Age Group'])

print (titanic_data)

