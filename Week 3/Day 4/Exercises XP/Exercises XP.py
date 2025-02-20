'''
Exercises XP

The importance of data visualization in data analysis.
How to use Python libraries such as Matplotlib and Seaborn for creating effective visualizations.

'''

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''
Exercise 1: Understanding Data Visualization

Explain why data visualization is important in data analysis.
Describe the purpose of a line graph in data visualization.

Data visualization is important in data analysis, because this enables the user to understand the data better, and to make better decisions.
Viewing the data in a visualized way.

Line graph:
Line plots are used to display quantitative values over a cetain value.

'''


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''

Exercise 2: Creating a Line Plot for Temperature Variation
Objective: Create a simple line plot using Matplotlib that represents temperature variations over a week.

'''

import matplotlib.pyplot as plt

days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
temps = [72, 74, 76, 80, 82, 78, 75]

plt.xlabel('days')
plt.ylabel('Temperature (°F)')

plt.title('Temperatures this week')

plt.plot(days, temps)

plt.show()


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''

Exercise 3: Visualizing Monthly Sales with a Bar Chart
Generate a bar chart using Matplotlib to visualize monthly sales data for a retail store.

'''

import matplotlib.pyplot as plt

monthly_sales = [5000, 5500, 6200, 7000, 7500]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']

plt.bar(months, monthly_sales)

plt.title('Sales Data')


plt.xlabel('Month')
plt.ylabel('Sales Amount ($)')
plt.show()


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''
Exercise 4: Visualizing the Distribution of CGPA

Create a histogram to visualize the distribution of students’ CGPA.
Dataset Overview: Assume the CGPA data is categorized into ranges and loaded in a DataFrame named df.

'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Let's assume we have a DataFrame with CGPA data
data = {'CGPA': [2.5, 3.0, 3.5, 4.0, 2.7, 3.8, 3.2, 3.9, 2.8, 3.4]}
df = pd.DataFrame(data)

# we will define bin edges, and labels for them
bins = [0, 3.5, 3.7, 4.0]
labels = ['Low', 'Medium', 'High']

# pd.cut() will categorize the CGPA values, and put it in a new column
df['CGPA level'] = pd.cut(df['CGPA'], bins=bins, labels=labels, right=False)

# print(df)  # DataFrame with CGPA Categories

# Create a histogram of the CGPA categories
sns.histplot(df['CGPA level'], kde=False, color='blue')

# Add title and labels
plt.title('Distribution of CGPA levels')
plt.xlabel('CGPA level')
plt.ylabel('Count')

# Display the plot
plt.show()



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''
Exercise 5: Comparing Anxiety Levels Across Different Genders

Use a bar plot to compare the proportion of students experiencing anxiety across different genders.
Dataset Overview: The dataset includes columns: ‘Do you have Anxiety?’ and ‘Choose your gender’.

Use Seaborn to create a bar plot comparing anxiety levels across genders from the dataset df.

'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Let's assume we have this sample of data:
data = {
    'Anxiety': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No'],
    'Genders': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female']
}

df = pd.DataFrame(data)

print("Table of data:")
print(df)

# Calculate the proportion of students experiencing anxiety by gender
anxiety_counts = df[df['Anxiety'] == 'Yes'].groupby('Genders').size()
total_counts = df.groupby('Genders').size()
proportions = anxiety_counts / total_counts

print("The proportion of students with anxiety - by gender:")
print(proportions)

# Create a bar plot
sns.barplot(x=proportions.index, y=proportions.values, palette='Set1')

# Customize the plot
plt.title('The proportion of students with anxiety - By gender')
plt.xlabel('Gender')
plt.ylabel('Proportion of students with anxiety')

# Display the plot
plt.show()


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''
Exercise 6: Exploring the Relationship Between Age and Panic Attacks

Create a scatter plot to explore the relationship between students’ age and the occurrence of panic attacks.
Dataset Overview: The dataset includes columns: ‘Age’ and ‘Do you have Panic Attacks?’.

Convert panic attack responses to numeric values (e.g., Yes=1, No=0).
Use Seaborn’s scatterplot to create a scatter plot with ‘Age’ on the x-axis and numeric panic attack responses on the y-axis.

'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Let's assume we have this DataFrame with Age and Panic Attacks
data = {
    'Age': [18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
    'Panic Attacks?': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

print("Table of data:")
print(df)

# Convert 'Panic Attacks?' to numeric values.  This will create a new column.
df['Panic Attacks'] = df['Panic Attacks?'].map({'Yes': 1, 'No': 0})

print("DataFrame with numeric panic attack data:")
print(df)

# Create a scatter plot
sns.scatterplot(data=df, x='Age', y='Panic Attacks', hue='Panic Attacks', palette='Set1')

# Customize the plot
plt.title('Showing the relationship')
plt.xlabel('Age')
plt.ylabel('Panic Attacks (1=Yes, 0=No)')

# Display the plot
plt.show()

