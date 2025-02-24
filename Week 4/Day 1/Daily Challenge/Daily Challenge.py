import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# randomly choosing temperature data between -5 and 35 degrees - for 10 cities - 12 months a year
temp_data_over_year = np.random.uniform(-5, 35, size=(10,12))

cities = ['city1','city2','city3','city4','city5','city6','city7','city8','city9','city10']
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

df = pd.DataFrame(temp_data_over_year, index=cities, columns=months)

# adding a column - the average of every row
df['Annual average temperature'] = df.mean(axis=1)

city_with_the_highest_average_temperature = df['Annual average temperature'].idxmax()
city_with_the_lowest_average_temperature = df['Annual average temperature'].idxmin()

# Plot each city's temperature over the months
plt.figure(figsize=(12, 8))
for city in df.index:
    plt.plot(df.columns, df.loc[city], marker='o', label=city)

plt.title('Monthly Temperatures for Each City')
plt.xlabel('Months')
plt.ylabel('Temperature')
plt.legend(title='City')
plt.grid(True)
plt.show()

