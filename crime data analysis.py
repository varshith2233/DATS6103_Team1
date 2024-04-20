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

# Number of Crimes by Area Name
plt.figure(figsize=(14,7))
sns.countplot(y = 'AREA NAME', data = data_2023, order = data_2023['AREA NAME'].value_counts().index)
plt.title('Number of Crimes by Area')
plt.xlabel('Number of Crimes')
plt.ylabel('Area Name')
plt.show()

#%%

#Victim Age Distribution by Age
plt.figure(figsize=(14, 7))
sns.boxplot(x='AREA NAME', y='Vict Age', data=data_2023)
plt.title('Victim Age Distribution by Area')
plt.xlabel('Area Name')
plt.xticks(rotation=90)
plt.ylabel('Victim Age')
plt.show()

#%%
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



# %%

# Major Crimes in Los Angeles

import plotly.express as px
# we are adding column 'count' to count how many times 'Crm Cd Desc' occurrence 
#%%
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
                   color='Hour')  # Optional: Remove 'color' if no color distinction is needed

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
### Data Modelling
data_2023.columns
data_2023.head()
#%%
data_2023["Weapon Used Cd"].replace({"Not Reported": 0}, inplace=True)
#data_2023["Weapon Used Cd"].unique()
#replacing: {"IC": 0, "AO": 1, "AA": 2, "JA": 3, "JO": 4, "CC": 5}
data_2023.Status.replace({"IC": 0, "AO": 1, "AA": 2, "JA": 3, "JO": 4, "CC": 5}, inplace=True)
#data_2023.Status.unique()

# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
#%%
X = data_2023[["AREA", "Vict Age", "Crm Cd","Premis Cd","Weapon Used Cd", 'Month']]
y = data_2023["Status"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()

# Fit and transform the scaler on the training data
X_train_scaled = scaler.fit_transform(X_train)
# Transform the test data using the same scaler
X_test_scaled = scaler.transform(X_test)
## Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

clf = DecisionTreeClassifier()

# Fit the classifier on the training data
clf.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
# %%
# Hyperparameter tuning for Decision Tree
for i in [2,3,4,5,6]:
    clf = DecisionTreeClassifier(max_depth=i)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with max depth {i}: {accuracy:.2f}")
# %%
from sklearn.neighbors import KNeighborsClassifier
#KNN
for i in [1,2,3,4,5]:
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with k = {i}: {accuracy:.2f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
# %%
# Random Forest
from sklearn.ensemble import RandomForestClassifier
# Initialize the Random Forest Classifier with 100 trees (you can change n_estimators as needed)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier on the scaled training data
rf_classifier.fit(X_train_scaled, y_train)

# Make predictions on the scaled test set
y_pred = rf_classifier.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
# %%
