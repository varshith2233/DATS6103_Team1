#%%

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
#%%

# Loading data

data = pd.read_csv('Crime_Data.csv')

#%%

# Display the first few rows of dataset

print(data.head())

print(data.info())

# Summary statistics

print(data.describe())

#%%

# Checking for number of unique data

print(data.nunique(),"\n\n")

# checking for missing values

print(data.isnull().sum())

#%%

# removing Unnecessary columns

columns_to_remove = ['Part 1-2', 'Mocodes', 'Vict Descent', 'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4']

data.drop(columns=columns_to_remove, inplace=True)

data.head()


# %%

# Removing Unnecessary data in Age column

data = data[data['Vict Age'] > 0]

data
#%%

# Converting 'DATE OCC' and 'Date Rptd' to datetime

data['DATE OCC'] = pd.to_datetime(data['DATE OCC'])
data['Date Rptd'] = pd.to_datetime(data['Date Rptd'])

# Taking only 2023 data from dataset

data_2023 = data[data['DATE OCC'].dt.year == 2023]

data_2023.head()
#%%
# Checking for missing values

missing_values = data_2023.isnull().sum()
print(missing_values)

# Removing missing values rows with small numbers

data_2023 = data_2023.dropna(subset = ['Vict Sex', 'Premis Cd','Crm Cd 1'])

# Fill 'Weapon Used Cd', 'Weapon Desc' with 'Not Reported' and 'Premis Desc' with 'unknown'

data_2023['Weapon Used Cd'].fillna('Not Reported', inplace = True)
data_2023['Weapon Desc'].fillna('Not Reported' , inplace = True)
data_2023['Premis Desc'].fillna('Unknown', inplace=True)

# Removing 'Cross Street' column with large number of missing values

data_2023 = data_2023.drop(columns = ['Cross Street'])


data_2023.head()


#%%
# Vict Age distribution

plt.figure(figsize = (10,6))
sns.histplot(data_2023['Vict Age'], bins = 30, kde = False)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Number of crimes')
plt.show()

# Victim Gender Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Vict Sex', data=data_2023)
plt.title('Victim Gender Distribution')
plt.show()

# Number of Crimes by Area Name
plt.figure(figsize=(14,7))
sns.countplot(y = 'AREA NAME', data = data_2023, order = data_2023['AREA NAME'].value_counts().index)
plt.title('Number of Crimes by Area')
plt.xlabel('Number of Crimes')
plt.ylabel('Area Name')
plt.show()

#Victim Age Distribution by Area
plt.figure(figsize=(14, 7))
sns.boxplot(x='AREA NAME', y='Vict Age', data=data_2023)
plt.title('Victim Age Distribution by Area')
plt.xlabel('Area Name')
plt.xticks(rotation=90)
plt.ylabel('Victim Age')
plt.show()

# Top 10 most common crime description 
plt.figure(figsize=(10, 8))
sns.countplot(y='Crm Cd Desc',data= data_2023, order=data_2023['Crm Cd Desc'].value_counts().iloc[:10].index)

plt.title('Top 10 Most Common Crime Descriptions')
plt.xlabel('Number of Occurrences')
plt.ylabel('Crime Description')
plt.show()

#Crime Trends over time
crimes_by_month = data_2023.resample('M', on='DATE OCC').size()

plt.figure(figsize=(12, 6))
crimes_by_month.plot()
plt.title('Crime Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Crimes')
plt.show()


#%%

# Crime Status Analysis
plt.figure(figsize=(10, 6))
sns.countplot(x='Status Desc', data=data_2023, order=data_2023['Status Desc'].value_counts().index)
plt.title('Crime Status Description')
plt.show()

#CRIME OCCURRENCE TIME BY AREA
plt.figure(figsize=(12, 8))
sns.boxplot(x='AREA', y='TIME OCC', data=data_2023)
plt.title('Crime Occurrence Time by Area')
plt.xlabel('Area Code')
plt.ylabel('Time of Crime (24-hour format)')
plt.show()
# %%
#Victim Age by  top 10 Crime Type

top_10_crime_types = data_2023['Crm Cd Desc'].value_counts().nlargest(10).index
filtered_data = data_2023[data_2023['Crm Cd Desc'].isin(top_10_crime_types)]

plt.figure(figsize=(12, 8))
sns.boxplot(data=filtered_data, x='Crm Cd Desc', y='Vict Age')
plt.title('Victim Age Distribution by Crime Type')
plt.xticks(rotation=90)
plt.xlabel('Crime Type')
plt.ylabel('Victim Age')

#Weapon Use in Crimes
plt.figure(figsize=(15, 8))
sns.countplot(y='Weapon Desc', data=data_2023, order=data_2023['Weapon Desc'].value_counts().head(10).index)
plt.title('Top 10 Weapons Used in Crimes')
plt.xlabel('Frequency')
plt.ylabel('Weapon Description')
plt.show()



# %%

# Major Crimes in Los Angeles

import plotly.express as px
# we are adding column 'count' to count how many times 'Crm Cd Desc' occurrence 
data_2023['count'] = 1

crime_counts = data_2023['Crm Cd Desc'].value_counts().reset_index()
crime_counts.columns = ['Crm Cd Desc', 'count']

fig = px.treemap(crime_counts, path=['Crm Cd Desc'], values='count', height=700,
                 title='Major Crimes in Los Angeles', color_discrete_sequence=px.colors.sequential.RdBu)
fig.data[0].textinfo = 'label+text+value'
fig.show()


# Top 10 Major Crimes 
top_crimes = crime_counts.head(10)

fig = px.bar(top_crimes, x='Crm Cd Desc', y='count', color='Crm Cd Desc', height=500,
             title='Top 10 Major Crimes in LA', labels={'Crm Cd Desc': 'Crime'})
fig.update_layout(bargap=0.2, bargroupgap=0.1, width = 1000 )
fig.show()

# %%
#Major crimes per month

import calendar
data_2023['Month'] = data_2023['DATE OCC'].dt.month
monthly_crime_counts = data_2023.groupby('Month').size().reset_index(name='Counts')
monthly_crime_counts = monthly_crime_counts.sort_values('Month')
fig = px.bar(monthly_crime_counts, x='Month', y='Counts', title='Major Crimes in LA per Month in 2023')
fig.update_layout(xaxis_title='Month', yaxis_title='Number of Crimes',
                  xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), ticktext=list(calendar.month_abbr)[1:]))
fig.show()

#%%
# Crime Count on each day
data_2023['Day_of_Week'] = data_2023['DATE OCC'].dt.day_name()
fig = px.histogram(data_2023, x='Day_of_Week', color='Day_of_Week')
fig.update_layout(
    title_text='Crime Count on Each Day of the Week', 
    xaxis_title_text='Day',
    yaxis_title_text='Crime Count', 
    bargap=0.2, 
    bargroupgap=0.1
)
fig.show()

#Percentage of Crimes by Day of week
crimes_per_day_of_week = data_2023['Day_of_Week'].value_counts().reset_index()
crimes_per_day_of_week.columns = ['Day_of_Week', 'Counts']

fig = px.pie(crimes_per_day_of_week, values='Counts', names='Day_of_Week',
             title='Percentage of Crimes by Day of the Week in LA for 2023')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()

#Crime Count on each hour
data_2023['Hour'] = data_2023['TIME OCC'].apply(lambda x: int(str(x).zfill(4)[:2]))

fig = px.histogram(data_2023, x='Hour',
                   title='Crime Count on Each Hour',
                   color='Hour') 

fig.update_layout(
    bargap=0.2,
    bargroupgap=0.1 
)
fig.show()


# %%
# Top 10 Crime count on each street
crime_counts_by_street = data_2023['AREA NAME'].value_counts().nlargest(10).reset_index()
crime_counts_by_street.columns = ['AREA NAME', 'Counts']


fig = px.bar(crime_counts_by_street, x='AREA NAME', y='Counts',
             title='Top 10 Crime Counts on Each Street',
             color='AREA NAME')

fig.show()


# %%

#Q3 Can we identify emerging spatial and temporal patterns or hotspots for crime categories to inform proactive and targeted interventions?

# Temporal Analysis
#Monthly Crime Trends

monthly_crimes = data_2023.groupby('Month').size()
plt.figure(figsize=(10, 6))
monthly_crimes.plot(kind='bar')
plt.title('Monthly Crime Trends in 2023')
plt.xlabel('Month')
plt.ylabel('Number of Crimes')
plt.xticks(rotation=0)
plt.show()

# From monthly crime trends chart, it appears the crime count is fairly consistent throughout the year, with slight variations.

# Hourly Crime Trends
hourly_crimes = data_2023.groupby('Hour').size()
plt.figure(figsize=(10, 6))
hourly_crimes.plot(kind='bar', color='orange')
plt.title('Hourly Crime Trends in 2023')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Crimes')
plt.xticks(rotation=0)
plt.show()

#%%

# Spatial Analysis
#The latitude and longitude data to identify clusters of high crime incidence

print(data_2023['LON'].describe())
print(data_2023['LAT'].describe())

# For longitude ('LON'), the max value is 0.0, which doesn't make sense given the mean and other percentiles are around -118. This suggests there may be data entry errors or missing values recorded as 0.0, which would not be valid longitude values for crime data presumably focused on a specific geographical area.
# For latitude ('LAT'), the minimum value is 0.0, which again is highly unlikely for a dataset focused on a specific city or region. This indicates there are likely invalid or missing entries recorded as 0.0.

# Filter out invalid longitude and latitude values
data_2023 = data_2023[(data_2023['LON'] != 0.0) & (data_2023['LAT'] != 0.0)]

# Consider min and max values of both longitude and latitide
longitude_bounds = [-118.66760, -118.27400]
latitude_bounds = [34.01650, 34.32900]   

data_2023 = data_2023[
    (data_2023['LON'].between(longitude_bounds[0], longitude_bounds[1])) &
    (data_2023['LAT'].between(latitude_bounds[0], latitude_bounds[1]))
]

# Plotting the data
plt.figure(figsize=(10, 10))
plt.scatter(data_2023['LON'], data_2023['LAT'], alpha=0.5)
plt.title('Spatial Distribution of Crimes')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()




# %%

#Clustering for Hotspot Detection
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import folium

coordinates = data_2023[['LAT', 'LON']]
# Scaling the coordinates to normalize data
scaler = StandardScaler()
coordinates_scaled = scaler.fit_transform(coordinates)
# Elbow method to find the optimal number of clusters
ssd = []
range_n_clusters = list(range(1, 11))
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=1000)
    kmeans.fit(coordinates_scaled)
    ssd.append(kmeans.inertia_)

# Plotting the results of the Elbow method
plt.figure(figsize=(10, 5))
plt.plot(range_n_clusters, ssd, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method For Optimal k')
plt.show()

#Looking at  graph, the elbow seems to be at around k=3. The curve starts to flatten out after that point, indicating that increasing the number of clusters beyond 3 yields diminishing returns in terms of reducing the SSD

# %%

# Fit K-Means with the optimal cluster number (assume k=3 from elbow plot observation)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(coordinates_scaled)
data_2023['cluster'] = kmeans.labels_



# Mapping
map = folium.Map(location=[data_2023['LAT'].mean(), data_2023['LON'].mean()], zoom_start=11)
colors = ['red', 'blue', 'green', 'purple', 'orange']

# Add points to the map
for idx, row in data_2023.iterrows():
    folium.CircleMarker([row['LAT'], row['LON']], radius=5, color=colors[row['cluster']]).add_to(map)

# Display map
map
# %%
# Choosing an optimal k from the elbow plot and applying K-Means clustering
optimal_k = 3 
kmeans = KMeans(n_clusters=optimal_k, max_iter=1000, random_state=42)
cluster_labels = kmeans.fit_predict(coordinates_scaled)

# Visualizing the clusters on a map
centroids = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids)  # Inverse the scaling to get actual coordinates

map_center = [data_2023['LAT'].mean(), data_2023['LON'].mean()]
map = folium.Map(location=map_center, zoom_start=12)

# Adding data points to the map
for idx, row in data_2023.iterrows():
    folium.CircleMarker(location=[row['LAT'], row['LON']],
                        radius=5,
                        color='blue',
                        fill=True,
                        fill_opacity=0.7).add_to(map)

# Adding centroids to the map
for centroid in centroids:
    folium.CircleMarker(location=[centroid[0], centroid[1]],
                        radius=10,
                        color='red',
                        fill=True,
                        fill_color='red',
                        fill_opacity=1).add_to(map)

# Displaying result
map
# %%
