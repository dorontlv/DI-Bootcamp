2.3.2025

Doron Fridman

Restaurants recomendation system
===========================================

I built this application as a project on the hackathon day.

This is an application that reads the restaurants and users data (a JSON file).

The file name is:
restaurants_data.json

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


There's a list of restaurants, with these columns:
"name"
"distance_from_city_center"
"cuisine_type"
"price_range"
"ambiance"

There's a list of users, with these columns:
"name"
"age"
"preferences"
"activity_log"

The "preferences" for each user are:
"liked_restaurants"
"liked_cuisines"
"activity_log"

The "activity_log" of each user has a:
"restaurant_name"
"date"
"rank"


The system loads the data from the disk, and then enables several operations:

logon_to_the_system()

find_restaurants_by_cuisine_type()
find_restaurants_by_distance()
find_restaurants_by_rank()
find_similar_liking()

plot_your_cuisine_type()
plot_your_rating_over_time()
plot_correlation_between_ranking_of_2_cuisine_types()

add_to_activity_log()


The plotting:
--------------

1.
A plot that shows your eating habits (cuisine types) over the years.

2.
A plot that shows your own average ranking over the years.

3.
This plot shows whether there is a correlation between the average ranking of 2 different cuisine types.
(Pearson correlation using scipy.stats.pearsonr)


Pay attension:
-------------------

1.
Sometimes, when displaying a tabular list of restaurants or users, there's an offset in the display.
This is a python bug.
It is not an issue in the program itself, and not an issue with the JSON data.


2.
When plotting the Pearson correlation plot (plot number 3).
You can choose 2 kinds of cuisines.
You will see if there's a correlation between the average ranking of these two cuisines.
Usually there's no correlation, off course, because this is just a synthetic data that was originally produced by the computer, and not by humans.
BUT:
If you will choose the following two cuisines: Italian and Sushi
Then you will see that there is a very high correlation !
This is because I deliberately faked the data of these two cuisines, so that we can see how it looks on the graph when there's a high correlation.
You can also see the result of the "Pearson Correlation Coefficient" and the "p-value".



