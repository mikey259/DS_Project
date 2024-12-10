import piplite
await piplite.install(['numpy'])
await piplite.install(['pandas'])
await piplite.install(['seaborn'])
# pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns

#First, let's read the SpaceX dataset into a Pandas dataframe and print its summary
from js import fetch
import io

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
resp = await fetch(URL)
dataset_part_2_csv = io.BytesIO((await resp.arrayBuffer()).to_py())
df=pd.read_csv(dataset_part_2_csv)
df.head(5)

#FlightNumber vs. PayloadMass
sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Pay load Mass (kg)",fontsize=20)
plt.show()

# Plot a scatter point chart with x axis to be Flight Number and y axis to be the launch site, and hue to be the class value
sns.catplot(y="LaunchSite", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Launch Site",fontsize=20)
plt.show()

# Plot a scatter point chart with x axis to be Pay Load Mass (kg) and y axis to be the launch site, and hue to be the class value
sns.scatterplot(y="LaunchSite", x="PayloadMass", hue="Class", data=df)
plt.xlabel("Pay load Mass (KG)",fontsize=20)
plt.ylabel("Launch Site",fontsize=20)
plt.show()

#relationship between success rate and orbit type
# HINT use groupby method on Orbit column and get the mean of Class column
# Group data by orbit and calculate success rate (mean of Class where Class == 'Success')
orbit_success_rate = df.groupby('Orbit')['Class'].mean().reset_index()
orbit_success_rate.rename(columns={'Class': 'Success_Rate'}, inplace=True)
sns.barplot(x="Orbit", y="Success_Rate", data=orbit_success_rate, hue="Success_Rate")
plt.xlabel("Orbit")
plt.ylabel("Success Rate")
plt.title("Success Rate of Missions by Orbit")
plt.show()

# Plot a scatter point chart with x axis to be FlightNumber and y axis to be the Orbit, and hue to be the class value
sns.scatterplot(y="Orbit", x="FlightNumber", hue="Class", data=df)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Orbit",fontsize=20)
plt.show()

# Plot a scatter point chart with x axis to be Payload Mass and y axis to be the Orbit, and hue to be the class value
sns.scatterplot(y="Orbit", x="PayloadMass", hue="Class", data=df)
plt.xlabel("Payload Mass (KG)",fontsize=20)
plt.ylabel("Orbit",fontsize=20)
plt.show()

#You can plot a line chart with x axis to be Year and y axis to be average success rate, to get the average launch success trend
#The function will help you get the year from the date:
# A function to Extract years from the date
year=[]
def Extract_year():
    for i in df["Date"]:
        year.append(i.split("-")[0])
    return year
Extract_year()
df['Date'] = year
df.head()

# Plot a line chart with x axis to be the extracted year and y axis to be the success rate
# Group by year and calculate success rate
success_rate_year = df.groupby('Date')['Class'].mean().reset_index()
success_rate_year.rename(columns={'Class': 'Success_Rate'}, inplace=True)
# Create the line chart
sns.lineplot(y="Success_Rate", x="Date", data=success_rate_year)
plt.xlabel("Year",fontsize=20)
plt.ylabel("Success_Rate",fontsize=20)
plt.title("Average Launch Success Rate by Year")
plt.show()

#By now, you should obtain some preliminary insights about how each important variable would affect the success rate,
#we will select the features that will be used in success prediction in the future module
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
features.head()

#Create dummy variables to categorical columns
# HINT: Use get_dummies() function on the categorical columns
features_one_hot = pd.get_dummies(features, columns=['Orbit', 'LaunchSite', 'LandingPad', 'Serial'])
features_one_hot.head()

#Cast all numeric columns to float64
# HINT: use astype function
features_one_hot = features_one_hot.astype('float64')

#We can now export it to a CSV
features_one_hot.to_csv('dataset_part_3.csv', index=False)
