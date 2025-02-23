'''

Mini-Project : Marketing Strategy

Mini-Project : Data Analysis for Marketing Strategy

Introduction
In this mini-project, we will perform data analysis to devise a marketing strategy based on various aspects like area analysis, customer analysis, product category analysis, and sales and profit time series.

Dataset
The US Superstore Dataset contains the following attributes:


'''

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

import pandas as pd

# Load the dataset
df = pd.read_excel('US Superstore data.xls')


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


'''
1.
Which states have the most sales?
'''

total_sales_by_state = df.groupby('State')['Sales'].sum().sort_values(ascending=False)

print(total_sales_by_state.head())



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''
2.
What is the difference between New York and California in terms of sales and profit?
(Compare the total sales and profit between New York and California.)

'''

# calculate the dif between the total sales of each state.
sales_dif = df[df['State']=='California']['Sales'].sum() - df[df['State']=='New York']['Sales'].sum()

if sales_dif > 0:
    print(f"The sales of California is larger by {sales_dif}.")
elif sales_dif < 0:
    print(f"The sales of New York is larger by {-sales_dif}.")
else:
    print(f"The sales of California and New York is equal.")

# We can also do the same calculation for the profit of each state.



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
'''

3.
Who is an outstanding customer in New York?
'''

# searching for the largest number in the Series and returning the index (the customer name)
print(df[df['State']=='New York'].groupby('Customer Name')['Sales'].sum().idxmax())



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''
4.
Are there any differences among states in profitability?

'''

# Because they are asking about the profitability (profit percentage), then you first need to calculate all the sales and all the profit of every state.
# And then look for the largest profitability.

# getting the total sales of every satate
table = df.groupby('State')['Sales'].sum()

table.name = 'Sales sum'  # rename it
table = table.to_frame()  # convert into a DataFrame

# getting the total profit of every satate
table['Profit sum'] = df.groupby('State')['Profit'].sum()

# get the max profitability (profit devided by sales)
print((table['Profit sum'] / table['Sales sum']).idxmax())


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''
5.
The Pareto Principle, also known as the 80/20 rule, is a concept derived from the work of Italian economist Vilfredo Pareto.
It states that roughly 80% of the effects come from 20% of the causes.
For instance, identifying the top 20% of products that generate 80% of sales or the top 20% of customers that contribute to 80% of profit can help in prioritizing efforts and resources.
This focus can lead to improved efficiency and effectiveness in business strategies.
Can we apply Pareto principle to customers and Profit ? (Determine if 20% of the customers contribute to 80% of the profit.)
'''

# I implemented this excercise, first by going through the lragest profit items (80%), and then I checked the customers (20%).

# the total number of customers in our data
total_number_of_customers = df['Customer Name'].drop_duplicates().size

# the total profit in our data
total_profit = df['Profit'].sum()

# take the table and sort it by profit in descending order
sorted_table = df.sort_values(by='Profit', ascending=False)

# print(sorted_table)

current_sum = 0
selected_rows = []  # a list of rows

# we are now iterating through the sorted table, and we stop when we get to 80% of the profit.
for index, row in sorted_table.iterrows():
    current_sum += row['Profit']
    selected_rows.append(row)
    if current_sum >= total_profit * 0.80:
        break

# create a new DataFrame with the selected rows
df_80_percent = pd.DataFrame(selected_rows)

# print(df_80_percent)

# how many customers are in the list of 80% of the profit
number_of_customers = df_80_percent['Customer Name'].drop_duplicates().size

if number_of_customers / total_number_of_customers <= 0.20:
    print("The Pareto Principle is kept - 20% of the customers contribute to 80% of profit")
else:
    print("The Pareto Principle in not kept")




print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''
6.
What are the Top 20 cities by Sales ?
What about the Top 20 cities by Profit ?
Are there any difference among cities in profitability ?
(Identify the top 20 cities based on total sales and total profit and analyze differences in profitability among these cities.)
'''

# getting the total sales of every city
table = df.groupby('City')['Sales'].sum()

table.name = 'Sales sum'  # rename the Series
table = table.to_frame()  # convert it into a DataFrame

# calculate and add the 'Profit sum' column to the table
table['Profit sum'] = df.groupby('City')['Profit'].sum()

# calculate the profitability (profit devided by sales).  Creatre a new column.
table['Profitability'] = table['Profit sum'] / table['Sales sum']

# sort the table, each time by a different column, and then display the first 20 rows of the table.
# diaplay only 2 columns.
table.sort_values('Sales sum', ascending=False)[['Sales sum']].head(20)
table.sort_values('Profit sum', ascending=False)[['Profit sum']].head(20)
top_20_profitability_cities = table.sort_values('Profitability', ascending=False)[['Profitability']].head(20)

# plotting
top_20_profitability_cities.plot(kind='bar', color='purple')
plt.title("Top 20 Cities by Profitability (Profit / Sales)")
plt.xlabel("City")
plt.ylabel("Profit as % of Sales")
plt.xticks(rotation=90)
plt.show()



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''
7.
What are the Top 20 customers by Sales?
'''

df.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False).head(20)


("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''
8.
Plot the Cumulative curve in Sales by Customers.
Can we apply Pareto principle to customers and Sales ?
'''

# getting the total sales of every customer
table_customers_sales = df.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False)

# convert into a DataFrame
table_customers_sales = table_customers_sales.to_frame()

# Calculate the cumulative sum.  Adding a column.
table_customers_sales['cumulative_sum_Sales'] = table_customers_sales['Sales'].cumsum()

import matplotlib.pyplot as plt

# Plot the cumulative curve
plt.plot(table_customers_sales['cumulative_sum_Sales'])
plt.xlabel('Customer Name')
plt.ylabel('cumulative_sum_Sales')
plt.title('Cumulative Curve')
plt.show()



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''
9.
Based on the analysis, make decisions on which states and cities to prioritize for marketing strategies.

'''

# We should prioritize for marketing strategies - the states and cities where the sales are the lowest.

table_states_sales = df.groupby('State')['Sales'].sum().sort_values().head()  # the lowest sales by states

table_cities_sales = df.groupby('City')['Sales'].sum().sort_values().head()  # the lowest sales by cities

