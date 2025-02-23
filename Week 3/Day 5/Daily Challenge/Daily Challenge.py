'''

Interactive Data Visualization...

Daily Challenge: Interactive Data Visualization with Matplotlib and Seaborn


'''

import pandas as pd

# Load the dataset
df = pd.read_excel('US Superstore data.xls')

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''
1. Data Preparation:
Download and explore the US Superstore data.
Perform basic data cleaning and preprocessing.
'''

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# No other kind of cleaning needs to be done.


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''
2. Data Visualization with Matplotlib:
Create an interactive line chart to show sales trends over the years.
Build an interactive map to visualize sales distribution by country.
'''

# pip install plotly

import pandas as pd
import plotly.express as px

# create a new column - extract the year from the 'Order Date'
df['Year'] = df['Order Date'].dt.year

# group by year - and calculate the total sales for every year
df_yearly_sales = df.groupby('Year')['Sales'].sum().reset_index()

# create an interactive line chart
fig = px.line(df_yearly_sales, x='Year', y='Sales', title='Sales trends over the years')

# Show the chart
fig.show()

fig = px.choropleth(
    df, 
    locations='Country',
    locationmode='country names',
    color='Sales',
    title='Sales distribution by country')

# Show the interactive map
fig.show()


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''
3. Data Visualization with Seaborn:
Use Seaborn to generate a bar chart showing top 10 products by sales.
Create a scatter plot to analyze the relationship between profit and discount.
'''

import seaborn as sns
import matplotlib.pyplot as plt

top_10_products_by_sales = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)

# convert into a DataFrame
top_10_products_by_sales = top_10_products_by_sales.to_frame()

# Create the bar chart
plt.figure(figsize=(12, 8))
sns.barplot(x='Sales', y='Product Name', data=top_10_products_by_sales)
plt.title('Top 10 products by sales')
plt.xlabel('Sales')
plt.ylabel('Product Name')
plt.show()

# Create the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Discount', y='Profit', data=df)
plt.title('Scatter plot - the relationship between profit and discount')
plt.xlabel('Discount')
plt.ylabel('Profit')
plt.show()

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


'''
4. Comparative Analysis:
Compare the insights gained from Matplotlib and Seaborn visualizations.
Document your observations about the ease of use and effectiveness of both tools.

Matplotlib is better for beginners.
It's highly flexible.

Seaborn has simplified syntax, and is specifically designed for statistical data visualization.
Its default themes are nicer.
It has built-in support for handling pandas DataFrames.  So it's easier to create plots directly from data.


'''

