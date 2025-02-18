'''

Exercises XP

Identify structured and unstructured data.
Convert unstructured data into structured formats.
Categorize and utilize different data types in a business context.

'''

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


'''
Exercise 1: Identifying Data Types

Below are various data sources.
Identify whether each one is an example of structured or unstructured data.

A company’s financial reports stored in an Excel file.      - This is a structured data.  Because it's an excel spreadsheet table.
Photographs uploaded to a social media platform.            - This is unstructured data.
A collection of news articles on a website.                 - This is unstructured data.
Inventory data in a relational database.                    - This is a structured data.  Because it's in a database - many files that have a structure.
Recorded interviews from a market research study.           - This is unstructured data.

'''

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''

Exercise 2: Transformation Exercise

For each of the following unstructured data sources, propose a method to convert it into structured data.
Explain your reasoning.

A series of blog posts about travel experiences.
On the blog website, add an option so that the user can choose the destination from a list, and rate the trip from an option list.

Audio recordings of customer service calls.
After every call, the customer support person should document the call.
He should choose a topic for the call, and fill in the customer details.

Handwritten notes from a brainstorming session.
The session manager should make a table:
Who is the person who proposed an idea.
What was the idea.
Who should implement it.

A video tutorial on cooking.
You should make a list of products.
For every product you should mention:
When it should be added to the cake.
What to do with it.

'''

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''

Exercise 3 : Import a file from Kaggle

Import the train dataset.
Use the train.csv file.
Print the first few rows of the DataFrame.

'''


import pandas as pd

data = pd.read_csv('train.csv')

print(data.head())


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''

Exercise 4: Importing a CSV File
Use the Iris Dataset CSV.

Download the Iris dataset CSV file and place it in the same directory as your Jupyter Notebook.
Import the CSV file using Pandas.
Display the first five rows of the dataset.

'''


import pandas as pd

data = pd.read_csv('Iris_dataset.csv')

print(data.head())


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

'''

Exercise 5 : Export a dataframe to excel format and JSON format.

Create a simple dataframe.
Export the dataframe to an excel file.
Export the dataframe to a JSON file.

'''


import pandas as pd

# Create a DataFrame
df = pd.DataFrame({
    'Column1': [1, 2, 3, 4, 5],
    'Column2': ['A', 'B', 'C', 'D', 'E'],
    'Column3': [True, False, True, False, True]
})

# Export to CSV
df.to_csv('my_dataframe.csv')

# result:
# Column1,Column2,Column3
# 1,A,True
# 2,B,False
# 3,C,True
# 4,D,False
# 5,E,True


# Export to JSON
df.to_json('my_dataframe.json')

# result:
# {"Column1":{"0":1,"1":2,"2":3,"3":4,"4":5},"Column2":{"0":"A","1":"B","2":"C","3":"D","4":"E"},"Column3":{"0":true,"1":false,"2":true,"3":false,"4":true}}



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


'''

Exercise 6: Reading JSON Data
Use a sample JSON dataset

Import the JSON data from the provided URL.
Use Pandas to read the JSON data.
Display the first five entries of the data.

'''

import pandas as pd

json_data = pd.read_json('posts.json')
print(json_data.head())

