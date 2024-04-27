#%%

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# Loading data
data = pd.read_csv('Crime_Data.csv')

# Display the first few rows of dataset
print(data.head())
print(data.info())
print(data.describe())

# Checking for number of unique data
print(data.nunique(),"\n\n")
print(data.isnull().sum())

# removing Unnecessary columns
columns_to_remove = ['Part 1-2', 'Mocodes', 'Vict Descent', 'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4']
data.drop(columns=columns_to_remove, inplace=True)

# Removing Unnecessary data in Age column
data = data[data['Vict Age'] > 0]
data['DATE OCC'] = pd.to_datetime(data['DATE OCC'])
data['Date Rptd'] = pd.to_datetime(data['Date Rptd'])

# Define post-pandemic period
post_pandemic_start = pd.Timestamp('2022-01-01')  # Assuming the pandemic ends by 2021

# Filter data for post-pandemic period
post_pandemic_data = data[data['Date Rptd'] >= post_pandemic_start]

# Extracting the last quarter of the post-pandemic period
post_pandemic_last_quarter_start = pd.Timestamp('2023-10-01')
post_pandemic_last_quarter_end = pd.Timestamp('2023-12-31')
post_pandemic_last_quarter_data = post_pandemic_data[(post_pandemic_data['Date Rptd'] >= post_pandemic_last_quarter_start) & 
                                                     (post_pandemic_data['Date Rptd'] <= post_pandemic_last_quarter_end)]

# Aggregate crime data by month for post-pandemic period
post_pandemic_monthly = post_pandemic_data.groupby(pd.Grouper(key='Date Rptd', freq='M')).size()
post_pandemic_last_quarter_monthly = post_pandemic_last_quarter_data.groupby(pd.Grouper(key='Date Rptd', freq='M')).size()

# Plotting crime trends over time for post-pandemic period
plt.figure(figsize=(12, 6))
plt.plot(post_pandemic_monthly.index, post_pandemic_monthly, label='Post-Pandemic')
plt.title('Monthly Crime Trends: Post-Pandemic')
plt.xlabel('Date')
plt.ylabel('Number of Crimes')
plt.legend()
plt.show()

# Analyze changes in overall crime rates in post-pandemic period
post_pandemic_total_crimes = post_pandemic_monthly.sum()
print(f"Total number of crimes in the post-pandemic period: {post_pandemic_total_crimes}")

# Analyze changes in crime types in post-pandemic period
post_pandemic_crime_types = post_pandemic_data['Crm Cd Desc'].value_counts(normalize=True)

# Compare the top 5 crime types in post-pandemic period
top5_post_pandemic_crime_types = post_pandemic_crime_types.head(5)
print("\nTop 5 Crime Types in the Post-Pandemic Period:")
print(top5_post_pandemic_crime_types)

# Analyze crime rates and patterns in the last quarter of the post-pandemic period
plt.figure(figsize=(12, 6))
plt.plot(post_pandemic_last_quarter_monthly.index, post_pandemic_last_quarter_monthly, label='Last Quarter of Post-Pandemic')
plt.title('Monthly Crime Trends: Last Quarter of Post-Pandemic Period')
plt.xlabel('Date')
plt.ylabel('Number of Crimes')
plt.legend()
plt.show()

# Analyze changes in crime types in the last quarter of the post-pandemic period
post_pandemic_last_quarter_crime_types = post_pandemic_last_quarter_data['Crm Cd Desc'].value_counts(normalize=True)
top5_post_pandemic_last_quarter_crime_types = post_pandemic_last_quarter_crime_types.head(5)
print("\nTop 5 Crime Types in the Last Quarter of Post-Pandemic Period:")
print(top5_post_pandemic_last_quarter_crime_types)

#%%
# Analyzing distribution of crime types in the last quarter of post-pandemic period
plt.figure(figsize=(10, 6))
post_pandemic_last_quarter_crime_types.value_counts().head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 Crime Types in the Last Quarter of Post-Pandemic Period')
plt.xlabel('Crime Type')
plt.ylabel('Number of Occurrences')
plt.xticks(rotation=45)
plt.show()

# Filter data for the last quarter of the post-pandemic period
last_quarter_post_pandemic_start = post_pandemic_start - pd.DateOffset(months=3)
last_quarter_post_pandemic_data = post_pandemic_data[(post_pandemic_data['DATE OCC'] >= last_quarter_post_pandemic_start) & (post_pandemic_data['DATE OCC'] < post_pandemic_start)]

# Calculate the distribution of crime types in the last quarter of the post-pandemic period
last_quarter_post_pandemic_crime_types = last_quarter_post_pandemic_data['Crm Cd Desc'].value_counts(normalize=True).head(10)

# Plotting the distribution of crime types
plt.figure(figsize=(10, 6))
sns.barplot(x=last_quarter_post_pandemic_crime_types.values, y=last_quarter_post_pandemic_crime_types.index, palette='viridis')
plt.title('Distribution of Crime Types in the Last Quarter of the Post-Pandemic Period')
plt.xlabel('Proportion of Crimes')
plt.ylabel('Crime Type')
plt.show()


# Calculate the distribution of crime types for the entire post-pandemic period
post_pandemic_crime_types = post_pandemic_data['Crm Cd Desc'].value_counts(normalize=True).head(10)

# Plotting the distribution of crime types
plt.figure(figsize=(10, 6))
sns.barplot(x=post_pandemic_crime_types.values, y=post_pandemic_crime_types.index, palette='viridis')
plt.title('Distribution of Crime Types in the Post-Pandemic Period')
plt.xlabel('Proportion of Crimes')
plt.ylabel('Crime Type')
plt.show()


