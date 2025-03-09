
import pandas as pd
import json
import random
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import pearsonr

# pip install tabulate
from tabulate import tabulate


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# global variables:
restaurants_df = None
users_df = None
activity_log_df = None
liked_restaurants_df = None
liked_cuisines_df = None

FILE_NAME = "restaurants_data.json"

own_name = None

'''
An example that will illustrate the JSON data structure we use:

{
    "restaurants": [
        {
            "name": "Pasta Palace",
            "distance_from_city_center": 3.5,
            "cuisine_type": "Italian",
            "price_range": "mid-range",
            "ambiance": "family-friendly"
        },
        {
            "name": "Sushi Sensation",
            "distance_from_city_center": 1.2,
            "cuisine_type": "Japanese",
            "price_range": "high-end",
            "ambiance": "romantic"
        }
    ]
    "users": [
        {
            "name": "John Doe",
            "age": 28,
            "preferences": {
                "liked_restaurants": ["Pasta Palace", "Burger Haven"],
                "liked_cuisines": ["Italian", "Mexican","Falafel"]
            },
            "activity_log": [
                {
                    "restaurant_name": "Pasta Palace",
                    "date": "2025-02-25",
                    "rank": 5
                },
                {
                    "restaurant_name": ""Sushi Sensation",
                    "date": "2025-01-08",
                    "rank": 4
                }
            ]
        }
    ]
}
'''


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def load_data_from_disk():

    global restaurants_df
    global users_df
    global activity_log_df
    global liked_restaurants_df
    global liked_cuisines_df


    print("\nHello.")
    print("Loading data...")

    # Load the JSON file
    data = json.load(open(FILE_NAME))

    # Normalize all JSON data into flat DataFrame tables

    # Restaurants:
    restaurants_df = pd.json_normalize(data["restaurants"])

    # Users:
    users_df = pd.json_normalize(data["users"])
    # Removing redundant columns
    users_df = users_df.drop("activity_log", axis=1)
    users_df = users_df.drop("preferences.liked_restaurants", axis=1)
    users_df = users_df.drop("preferences.liked_cuisines", axis=1)

    # Activity log:
    activity_log_df = pd.json_normalize(data["users"], record_path="activity_log", meta=["name", "age"])

    # Preferences:
    # Extract the 2 preferences into 2 separate DataFrames
    liked_restaurants_df = pd.json_normalize(data["users"], record_path=["preferences", "liked_restaurants"], meta=["name"])
    liked_cuisines_df = pd.json_normalize(data["users"], record_path=["preferences", "liked_cuisines"], meta=["name"])
    # Rename columns for clarity
    liked_restaurants_df.columns = ["liked_restaurant", "name"]
    liked_cuisines_df.columns = ["liked_cuisine", "name"]



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def get_user_menu_choice():

    print("\nPlease choose an option from the menu.  (Press ENTER to continue)")
    input()

    print("")
    print ("  MENU:")
    print ("==============================================================")
    print("|  ")
    print ("|  FIND:")
    print ("|  ---------")
    print ("|  (C) Find: prefered restaurant by CUISINE TYPE")
    print ("|  (D) Find: prefered restaurant by DISTANCE from city center")
    print ("|  (R) Find: restaurants that got high RANKING by other diners")
    print ("|  (S) Find: SIMILAR restaurants that other people liked")
    print("|  ")
    print ("|  PLOT:")
    print ("|  ---------")
    print ("|  (1) Plot: the distribution of your own cuisine type eating along the years")
    print ("|  (2) Plot: visualize your average ranking along the years")
    print ("|  (3) Plot: correlation between ranking of 2 cuisine types")
    print("|  ")
    print ("|  MANAGE:")
    print ("|  ---------")
    print ("|  (A) Activity log: add a visit to a restaurant and rank it")
    print ("|  (L) Log-on as different user")
    print ("|  (Q) *** QUIT ***")
    print("|  ")
    
    return input("\nWhat do you want to do ? ").lower()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def find_restaurants_by_cuisine_type():
    
    # get the cuisine types from the restaurants df
    cuisine_type_df = restaurants_df["cuisine_type"].drop_duplicates()
    cuisine_type_list = list(cuisine_type_df)
    # Print the cuisine types
    print("\nThese are the cuisine types we have:")
    print("---------------------------------------")
    for item in cuisine_type_list:
        print(item)
    
    cuisine_type = input("\nEnter your choice for cuisine type: ").lower()
    print("")
    
    # get only the restaurants that belong to this cuisine type
    df = restaurants_df[restaurants_df["cuisine_type"].str.lower() == cuisine_type]

    if df.empty:
        print("No restaurants meet this criteria.")
        return
    
    # drop this column from the display, since this was the criteria column
    df = df.drop("cuisine_type", axis=1)

    df = df.sort_values(by="name")
    
    print(f"Restaurants that server {cuisine_type} cuisine are:")

    # print the DataFrame without the index row using tabulate
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# a helper function - checks if a string is a float
def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def find_restaurants_by_distance():

    print("")

    # the user will input the distance number
    while True:
        distance = input("Enter the distance you are willing to go for good food: ")
        if distance and (is_float(distance)):
            break

    distance = float(distance)
    
    print("")
    
    # get restaurants with no more than such distance
    df = restaurants_df[restaurants_df["distance_from_city_center"] <= distance]

    if df.empty:
        print("No restaurants meet this criteria.")
        return
    
    # sort by distance
    df = df.sort_values(by="distance_from_city_center")
    
    print(f"Restaurants that are closer than {distance} kilometers from city center are:")
    print("(For you convenience, the list is sorted by driving distance)")

    # print the DataFrame without the index row using tabulate
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def find_restaurants_by_rank():

    global own_name

    # check the activity log for high ranking of other users.
    df = activity_log_df[(activity_log_df["rank"] >= 4) & (activity_log_df["name"] != own_name)]

    if df.empty:
        print("No restaurants meet this criteria.")
        return
    
    # get just the restaurant names
    df = df["restaurant_name"].to_frame().drop_duplicates()
    df = df.sort_values(by="restaurant_name")
    high_rank_restaurants = list(df["restaurant_name"])
    
    print(f"\nRestaurants that got high ranking (4 or 5) by other diners along the years are:")
    print("------------------------------------------------------------------------------------\n")

    num_columns = 3
    for i in range(0, len(high_rank_restaurants), num_columns):
        print('\t'.join(high_rank_restaurants[i:i+num_columns]))



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def find_similar_liking():
    global own_name


    # all restaurants I like
    df = liked_restaurants_df[liked_restaurants_df["name"] == own_name]

    # randomly choose one of the restaurants I like
    one_restaurant_I_like = random.choice(list(df["liked_restaurant"]))

    # other people who also liked this restaurant
    df = liked_restaurants_df[(liked_restaurants_df["name"] != own_name) & (liked_restaurants_df["liked_restaurant"] == one_restaurant_I_like)]

    if df.empty:
        print(f"\nYou liked eating at {one_restaurant_I_like}, but no other people liked this restaurant.")
        return

    other_people = list(df["name"])  # a list of other people who also liked this restaurant

    # all the restaurants that these other people liked
    df = liked_restaurants_df[liked_restaurants_df["name"].isin(other_people)]
    
    # but need to omit the restaurant that I liked
    df = df[df["liked_restaurant"] != one_restaurant_I_like]

    df = df["liked_restaurant"].to_frame().drop_duplicates()
    df = df.sort_values(by="liked_restaurant")
    other_liked_restaurants = list(df["liked_restaurant"])

    print(f"\nWe saw that you liked eating at {one_restaurant_I_like}, for example.")
    print("We examined the taste of other diners that also liked this restaurant.")
    print("Considering their preferences, we think that you should try these restaurants as well:")
    print("-----------------------------------------------------------------------------------------")

    num_columns = 2
    for i in range(0, len(other_liked_restaurants), num_columns):
        print('\t'.join(other_liked_restaurants[i:i+num_columns]))



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# Plot: the distribution of your own cuisine type eating along the years
def plot_your_cuisine_type():
    
    global own_name

    df = activity_log_df[(activity_log_df["name"] == own_name)]  # get my own activity log

    # Iterate through rows - to retrieve the cuisine type from the restaurants df
    for index, row in df.iterrows():
        retrieved_cuisine_type = restaurants_df.loc[restaurants_df['name'] == row["restaurant_name"], 'cuisine_type'].iloc[0]
        df.loc[index,"retrieved_cuisine_type"] = retrieved_cuisine_type

    # get how many restaurants there are in each cuisine type
    retrieved_cuisine_type_df = df.groupby("retrieved_cuisine_type").size()

    # Plotting the bar graph
    plt.figure(figsize=(10, 5))
    retrieved_cuisine_type_df.plot(kind='bar', legend=False)
    plt.title('The distribution of your own cuisine type eating along the years')
    plt.xlabel('Cuisine Type')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# a helper function - extract the year+month from a date
def extract_year_month(date_str):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    return date_obj.strftime("%Y-%m")


# Plot: Visualize your average ratings over time
def plot_your_rating_over_time():
    
    global own_name

    df = activity_log_df[(activity_log_df["name"] == own_name)]  # get my own activity log

    # add a column with the year-month
    df["year_month"] = df["date"].apply(extract_year_month)

    # get the average rank in every month
    average_df = df.groupby("year_month")["rank"].mean().reset_index()

    # rename columns
    average_df.columns = ['year_month', 'average']

    average_df = average_df.sort_values(by="year_month")

    # Plotting the line graph
    plt.figure(figsize=(10, 5))
    plt.plot(average_df['year_month'], average_df['average'], marker='o', linewidth=4)
    plt.title('Your average ratings for each month over the years')
    plt.xlabel('Month')
    plt.ylabel('Average ranking')
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.show()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def plot_correlation_between_ranking_of_2_cuisine_types():

    
    # choose 2 cuisine types to compare between

    # Print the cuisine types
    cuisine_type_df = restaurants_df["cuisine_type"].drop_duplicates()
    cuisine_type_list = list(cuisine_type_df)
    lowercase_list = [s.lower() for s in cuisine_type_list]
    print("\nThese are the cuisine types we have:")
    print("---------------------------------------")
    for item in cuisine_type_list:
        print(item)

    # the user will input a cuisine type
    while True:
        first_cuisine = input("\n\nPlease enter a cuisine type: ")
        if first_cuisine and (first_cuisine.lower() in lowercase_list):
            break
        else:
            print("This cuisine is not on the list.  Please choose again.")

    # the user will input a second cuisine type
    while True:
        second_cuisine = input("\nPlease enter a second cuisine type: ")
        if second_cuisine and (second_cuisine.lower() in lowercase_list):
            if first_cuisine.lower() != second_cuisine.lower():
                break
            else:
                print("You must choose two different cuisines.  Please choose again.")
        else:
            print("This cuisine is not on the list.  Please choose again.")


    # get the cuisine types directly from the list, even if the user entered them in a different letter case
    first_cuisine = [s for s in cuisine_type_list if first_cuisine.lower() in s.lower()][0]
    second_cuisine = [s for s in cuisine_type_list if second_cuisine.lower() in s.lower()][0]
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~

    # make a copy of the DataFrame
    df = activity_log_df.copy()

    print("\nRetrieving cuisine types for the entire activity log.")
    print("Please wait...")

    # Iterate through rows - to retrieve the cuisine type from the restaurants df
    for index, row in df.iterrows():
        retrieved_cuisine_type = restaurants_df.loc[restaurants_df['name'] == row["restaurant_name"], 'cuisine_type'].iloc[0]
        df.loc[index,"retrieved_cuisine_type"] = retrieved_cuisine_type
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~
    
    '''
    When users rank the same cuisine type multiple times, we need to calculate the average ranking for each user and cuisine type before pivoting the DataFrame.

    '''

    # calculate the average ranking for each user and cuisine type
    user_avg_ranks = df.groupby(['name', 'retrieved_cuisine_type'])['rank'].mean().reset_index()

    # pivot the DataFrame to create a new DataFrame with paired data
    pivot_df = user_avg_ranks.pivot(index='name', columns='retrieved_cuisine_type', values='rank')

    # take only the two cuisines, and filter out users who did not rank both cuisine types (drop NaN values)
    filtered_df = pivot_df[[first_cuisine, second_cuisine]].dropna()
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~

    # get the paired rankings from the df
    first_cuisine_ranks = filtered_df[first_cuisine]
    second_cuisine_ranks = filtered_df[second_cuisine]

    # perform a Pearson correlation test
    corr, p_value = pearsonr(first_cuisine_ranks, second_cuisine_ranks)

    print("\nThe statistics:")
    print("Pearson Correlation Coefficient:", corr)
    print("p-value:", p_value)

    # Create a scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(first_cuisine_ranks, second_cuisine_ranks, c='blue', marker='o')
    plt.title(f'Scatter plot of {first_cuisine} cuisine vs. {second_cuisine} cuisine rankings')
    plt.xlabel(f'{first_cuisine} cuisine rank')
    plt.ylabel(f'{second_cuisine} cuisine rank')
    plt.grid(True)
    plt.show()



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def logon_to_the_system():

    global own_name

    # get all users list
    users_list = list(users_df["name"])
    lowercase_list = [s.lower() for s in users_list]

    # print the list of users

    print("\nWe have these people registered on our restaurant system:")
    print("------------------------------------------------------------\n")
    
    num_columns = 4
    for i in range(0, len(users_list), num_columns):
        print('\t\t\t\t\t'.join(users_list[i:i+num_columns]))


    # the user will input a log-on name
    while True:
        own_name = input("\nPlease enter your log-on name: ")
        if own_name and (own_name.lower() in lowercase_list):
            break
        else:
            print("This name is not on the list.  Please enter your name again.")


    # get the user name directly from the list, even if the user entered it in a different letter case
    own_name = [s for s in users_list if own_name.lower() in s.lower()][0]

    print(f"\nWelcome {own_name}.")
    


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# a helper function - checking the date format
def is_valid_date(date_str):
    try:
        # checking for a date string
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False
    

def add_to_activity_log():

    # get all restaurant names
    restaurants_names = list(restaurants_df["name"].sort_values())

    # print the list of restaurants
    print(f"\nRestaurants we have in our database:")
    print("---------------------------------------\n")

    numbered_restaurants_names = []
    for index, restaurant in enumerate(restaurants_names, start=1):
        numbered_restaurants_names.append(f"{index}. {restaurant}")

    num_columns = 2
    for i in range(0, len(numbered_restaurants_names), num_columns):
        print('\t\t'.join(numbered_restaurants_names[i:i+num_columns]))

    print ("\n")


    # let the user enter a number to choose a restaurant from the list
    while True:
        try:
            chosen_restaurant = int(input("Enter the number of the restaurant you want to choose: "))
            if not(1 <= chosen_restaurant <= len(restaurants_names)):
                print("Invalid choice. Please enter a number from the list.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    print("")


    # the user will enter a visit date to this restaurant
    while True:
        try:
            chosen_date = input("On which date did you visit that restaurant (YYYY-MM-DD): ")
            if not(is_valid_date(chosen_date)):
                print("Invalid date. Please enter a valid date value.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a valid date.")

    print("")


    # the user will enter a rank for this restaurant
    while True:
        try:
            # ask the user to enter a number
            chosen_rank = int(input("How would you rank this restaurant ? (between 1 and 5): "))
            # validate the number
            if 1 <= chosen_rank <= 5:
                break
            else:
                print("Invalid input. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    
    print("")

    # retrieve the user's age
    retrieved_user_age = users_df.loc[users_df['name'] == own_name, 'age'].iloc[0]

    # add a new row to the activity log df
    activity_log_df.loc[len(activity_log_df)] = [restaurants_names[chosen_restaurant-1], chosen_date, chosen_rank, own_name, retrieved_user_age]

    print("Your visit to this restaurant has been registered in your activity log.")



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



def main():

    global own_name

    load_data_from_disk()

    logon_to_the_system()

    while True:

        while True:        
            selection = get_user_menu_choice()
            if selection and (selection in "123csrdqla"):
                break
            
        if selection == 'q':
            print("")
            print ("Quiting the program.\n\n")
            break

        # add a restaurant to the activity log
        if selection == 'a':
            add_to_activity_log()

        # log-on
        if selection == 'l':
            logon_to_the_system()

        # Filtering restaurants from a dataset that match the user’s preferences.
        if selection == 'c':
            find_restaurants_by_cuisine_type()

        # Filtering restaurants by distance
        if selection == 'd':
            find_restaurants_by_distance()

        # Filtering restaurants by rank
        if selection == 'r':
            find_restaurants_by_rank()

        # Find similar restaurants by others
        if selection == 's':
            find_similar_liking()

        # Plot your cuisine type
        if selection == '1':
            plot_your_cuisine_type()

        # Plot your rating over time
        if selection == '2':
            plot_your_rating_over_time()

        # Plot correlation between ranking of 2 cuisine types
        if selection == '3':
            plot_correlation_between_ranking_of_2_cuisine_types()






main()


