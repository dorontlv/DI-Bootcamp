'''

Data Handling and Analysis in Python

Daily Challenge: Data Handling and Analysis in Python


What You Will Learn
Advanced techniques for data normalization, reduction, and aggregation.
Skills in gathering, exploring, integrating, and cleaning data using Python.
Proficiency in using Pandas for complex data manipulation.


Your Task
Download and import the Data Science Job Salary dataset.
Normalize the ‘salary’ column using Min-Max normalization which scales all salary values between 0 and 1.
Implement dimensionality reduction like Principal Component Analysis (PCA) or t-SNE to reduce the number of features (columns) in the dataset.
Group the dataset by the ‘experience_level’ column and calculate the average and median salary for each experience level (e.g., Junior, Mid-level, Senior).

'''

import pandas as pd

# Load the Titanic dataset
df_salaries = pd.read_csv('datascience_salaries.csv')

print (df_salaries)

# ~~~~~~~~

# Normalize the ‘salary’ column using Min-Max normalization which scales all salary values between 0 and 1.

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# adding a column - where the salary is normalized
df_salaries['salary_normalized'] = scaler.fit_transform(df_salaries[['salary']])

print (df_salaries)


# ~~~~~~~~

# implement dimensionality reduction like Principal Component Analysis (PCA) or t-SNE to reduce the number of features (columns) in the dataset.

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

# including only numerical columns.  but excluding the column 'Unnamed: 0' because it's just an index number.
pca_components = pca.fit_transform(df_salaries.select_dtypes(include=[float, int]).drop('Unnamed: 0', axis=1))

# print (pca_components)

df_pca = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])

print(df_pca)

# ~~~~~~~~

# Group the dataset by the ‘experience_level’ column and calculate the average and median salary for each experience level (e.g., Junior, Mid-level, Senior).

grouped_df = df_salaries.groupby('experience_level')['salary'].agg(['mean', 'median'])

print(grouped_df)



