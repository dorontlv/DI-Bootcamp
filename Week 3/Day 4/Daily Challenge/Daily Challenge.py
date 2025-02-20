'''

Exploring the World Happiness Report

use Matplotlib alongside pandas to analyze
visualize data from the World Happiness Report
focus on aspects like happiness scores, economic factors, and regional differences.

Your objective is to delve into the World Happiness Report dataset, which includes columns like Country, Year, Happiness_Score, GDP_per_Capita, Social_Support, Healthy_Life_Expectancy, Freedom_to_Make_Life_Choices, Generosity, and Perceptions_of_Corruption.

Address missing values and adjust data types as necessary.

Employ a Matplotlib scatter plot to investigate the relationship between ‘Social support’ and ‘Score’.

Create a Matplotlib subplot that compares ‘GDP per Capita’ and ‘Healthy Life Expectancy’ across different regions.
Use a bar plot for ‘GDP per Capita’ and a line plot for ‘Healthy Life Expectancy’ on the same axes to observe how economic strength relates to health outcomes in different regions.

'''


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
df = pd.read_csv('world-happiness-2019.csv')

# Address missing values
df = df.dropna()

# there's no need for data types adjustment

# Scatter plot for "Social support" and "Score"
sns.scatterplot(data=df, x='Social support', y='Score')
plt.title('The relationship between social support and happiness score')
plt.xlabel('Social support')
plt.ylabel('Happiness score')
plt.show()

# ~~~~~~

# Subplot for GDP per capita and Healthy life expectancy
fig, ax1 = plt.subplots(figsize=(12, 8))

# barplot
sns.barplot(data=df, x='Country or region', y='GDP per capita', ax=ax1, palette='Blues')
ax1.set_ylabel('GDP per capita')
ax1.set_xlabel('Country or region')

ax2 = ax1.twinx()  # Create a twin Axes sharing the same x-axis.

# lineplot
sns.lineplot(data=df, x='Country or region', y='Healthy life expectancy', ax=ax2, color='r', marker='o')
ax2.set_ylabel('Healthy life expectancy')

plt.title('Comparing GDP and Health')

plt.show()



