import piplite
await piplite.install(['folium'])
await piplite.install(['pandas'])
import folium
import pandas as pd
# Import folium MarkerCluster plugin
from folium.plugins import MarkerCluster
# Import folium MousePosition plugin
from folium.plugins import MousePosition
# Import folium DivIcon plugin
from folium.features import DivIcon

## Task 1: Mark all launch sites on a map
#First, let's try to add each site's location on a map using site's latitude and longitude coordinates
#The following dataset with the name spacex_launch_geo.csv is an augmented dataset with latitude and longitude added for each site.

# Download and read the `spacex_launch_geo.csv`
from js import fetch
import io

URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv'
resp = await fetch(URL)
spacex_csv_file = io.BytesIO((await resp.arrayBuffer()).to_py())
spacex_df=pd.read_csv(spacex_csv_file)

#Now, you can take a look at what are the coordinates for each site.
# Select relevant sub-columns: `Launch Site`, `Lat(Latitude)`, `Long(Longitude)`, `class`
spacex_df = spacex_df[['Launch Site', 'Lat', 'Long', 'class']]
launch_sites_df = spacex_df.groupby(['Launch Site'], as_index=False).first()
launch_sites_df = launch_sites_df[['Launch Site', 'Lat', 'Long']]
for index, row in launch_sites_df.iterrows():
    print(index, row)

#We first need to create a folium Map object, with an initial center location to be NASA Johnson Space Center at Houston, Texas.
# Start location is NASA Johnson Space Center
nasa_coordinate = [29.559684888503615, -95.0830971930759]
site_map = folium.Map(location=nasa_coordinate, zoom_start=10)

#We could use folium.Circle to add a highlighted circle area with a text label on a specific coordinate. For example,
# Create a blue circle at NASA Johnson Space Center's coordinate with a popup label showing its name
circle = folium.Circle(nasa_coordinate, radius=1000, color='#d35400', fill=True).add_child(folium.Popup('NASA Johnson Space Center'))
# Create a blue circle at NASA Johnson Space Center's coordinate with a icon showing its name
marker = folium.map.Marker(
    nasa_coordinate,
    # Create an icon as a text label
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % 'NASA JSC',
        )
    )
site_map.add_child(circle)
site_map.add_child(marker)

#TODO: Create and add folium.Circle and folium.Marker for each launch site on the site map
# Initialise the map
site_map = folium.Map(location=nasa_coordinate, zoom_start=5)
# For each launch site, add a Circle object based on its coordinate (Lat, Long) values. In addition, add Launch site name as a popup label
# Loop through each launch site in the DataFrame
for index, row in launch_sites_df.iterrows():
    launch_site = row['Launch Site']
    latitude = row['Lat']
    longitude = row['Long']
    coordinates = [latitude, longitude]
    circle = folium.Circle(coordinates, radius=1000, color='#000000', fill=True).add_child(folium.Popup(f"Launch Site: {launch_site}"))
    # Add the circle to the map
    circle.add_to(site_map)
    marker = folium.map.Marker(coordinates, icon=DivIcon(icon_size=(20,20),icon_anchor=(0,0), html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % launch_site, ))
    marker.add_to(site_map)
site_map


# Task 2: Mark the success/failed launches for each site on the map
spacex_df.head(5)
#Next, let's create markers for all launch records. If a launch was successful (class=1), then we use a green marker and if a launch was failed, we use a red marker (class=0)
#Note that a launch only happens in one of the four launch sites, which means many launch records will have the exact same coordinate. Marker clusters can be a good way to simplify a map containing many markers having the same coordinate.
#Let's first create a MarkerCluster object
marker_cluster = MarkerCluster()
#TODO: Create a new column in spacex_df dataframe called marker_color to store the marker colors based on the class value
# Apply a function to check the value of `class` column
# If class=1, marker_color value will be green
# If class=0, marker_color value will be red
# Create a new column 'marker_color' based on the 'class' column
spacex_df['marker_color'] = spacex_df['class'].apply(lambda x: 'green' if x == 1 else 'red')#lambda is a 'throwaway or short-term use function for small functions'

#TODO: For each launch result in spacex_df data frame, add a folium.Marker to marker_cluster
# Add marker_cluster to current site_map
site_map.add_child(marker_cluster)

# for each row in spacex_df data frame
# create a Marker object with its coordinate
# and customize the Marker's icon property to indicate if this launch was successed or failed,
# e.g., icon=folium.Icon(color='white', icon_color=row['marker_color']
for index, row in spacex_df.iterrows():
    # TODO: Create and add a Marker cluster to the site map
    # marker = folium.Marker(...)
    latitude = row['Lat']
    longitude = row['Long']
    marker_color = row['marker_color']

    marker = folium.Marker(
        location=[latitude, longitude],
        icon=folium.Icon(color=marker_color),
        popup=f"Launch Site: {row['Launch Site']}<br>Class: {row['class']}"
    )
    marker_cluster.add_child(marker)
# Add the marker cluster to the map
marker_cluster.add_to(site_map)
site_map

##### TASK 3: Calculate the distances between a launch site to its proximities ########

#'Let's first add a MousePosition on the map to get coordinate for a mouse over a point on the map. As such, while you are exploring the map, you can easily find the coordinates of any points of interests (such as railway)'
# Add Mouse Position to get the coordinate (Lat, Long) for a mouse over on the map
formatter = "function(num) {return L.Util.formatNum(num, 5);};"
mouse_position = MousePosition(
    position='topright',
    separator=' Long: ',
    empty_string='NaN',
    lng_first=False,
    num_digits=20,
    prefix='Lat:',
    lat_formatter=formatter,
    lng_formatter=formatter,
)

site_map.add_child(mouse_position)
site_map

#Now zoom in to a launch site and explore its proximity to see if you can easily find any railway, highway, coastline, etc. Move your mouse to these points and mark down their coordinates (shown on the top-left) in order to the distance to the launch site.
from math import sin, cos, sqrt, atan2, radians

def calculate_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

###TODO: Mark down a point on the closest coastline using MousePosition and calculate the distance between the coastline point and the launch site.
# Get coordinates of the closest coastline point using MousePosition
coastline_lat = 28.56246
coastline_lon = -80.56783
launch_site_lat = spacex_df['Lat'][1]
launch_site_lon = spacex_df['Long'][1]

# Calculate the distance
distance_coastline = calculate_distance(launch_site_lat, launch_site_lon, coastline_lat, coastline_lon)

# Create a marker to display the distance
distance_marker = folium.Marker(
    [coastline_lat, coastline_lon],
    icon=folium.DivIcon(
        icon_size=(20, 20),
        icon_anchor=(0, 0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_coastline),
    )
)

# Add the marker to the map
site_map.add_child(distance_marker)
# Display the map
site_map


# Create a `folium.PolyLine` object using the coastline coordinates and launch site coordinate
# lines=folium.PolyLine(locations=coordinates, weight=1)
coordinates = [(launch_site_lat, launch_site_lon), (coastline_lat, coastline_lon)]
lines = folium.PolyLine(locations=coordinates, weight=2, color='blue')
site_map.add_child(lines)

###TODO: Similarly, you can draw a line betwee a launch site to its closest city, railway, highway, etc. You need to use MousePosition to find the their coordinates on the map first
