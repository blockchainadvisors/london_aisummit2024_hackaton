import numpy as np
import pandas as pd
import plotly.graph_objects as go
from flask import Flask, render_template
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from dotenv import load_dotenv

import os

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()
MAPBOX_ACCESS_TOKEN = os.getenv('MAPBOX_ACCESS_TOKEN')

def constrained_kmeans(data, max_points_per_cluster=20, max_distance=None):
    def fit_kmeans(data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)
        return kmeans

    def split_large_clusters(points, max_points_per_cluster):
        n_sub_clusters = (len(points) // max_points_per_cluster) + 1
        sub_kmeans = fit_kmeans(np.array(points), n_sub_clusters)
        sub_clusters = []
        for sub_label in np.unique(sub_kmeans.labels_):
            sub_cluster_points = np.array(points)[sub_kmeans.labels_ == sub_label]
            sub_clusters.append(sub_cluster_points)
        return sub_clusters

    def filter_clusters(data, labels, max_points_per_cluster, max_distance):
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(data[i])
        
        filtered_clusters = []
        for points in clusters.values():
            if len(points) > max_points_per_cluster:
                sub_clusters = split_large_clusters(points, max_points_per_cluster)
                filtered_clusters.extend(sub_clusters)
            else:
                filtered_clusters.append(np.array(points))
        
        if max_distance is not None:
            final_clusters = []
            for cluster in filtered_clusters:
                centroid = np.mean(cluster, axis=0)
                distances = cdist(cluster, [centroid])
                within_distance = cluster[distances.flatten() <= max_distance]
                final_clusters.append(within_distance)
            return final_clusters
        return filtered_clusters

    n_initial_clusters = max(1, len(data) // max_points_per_cluster)
    initial_kmeans = fit_kmeans(data, n_initial_clusters)
    filtered_clusters = filter_clusters(data, initial_kmeans.labels_, max_points_per_cluster, max_distance)
    return filtered_clusters

def calculate_initial_zoom(lat_range, lon_range):
    # This function estimates an appropriate zoom level for the given latitude and longitude range.
    max_range = max(lat_range, lon_range)
    zoom = np.log2(360 / max_range) - 1
    return zoom

def generate_plot(clusters, data_bounds):
    fig = go.Figure()
    
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        fig.add_trace(go.Scattermapbox(
            lon=cluster[:, 1], lat=cluster[:, 0],
            mode='markers',
            marker=dict(size=10),
            name=f'Cluster {i+1}'
        ))

    # Calculate the margin
    lat_margin = (data_bounds['lat_max'] - data_bounds['lat_min']) * 0.10
    lon_margin = (data_bounds['lon_max'] - data_bounds['lon_min']) * 0.10
    
    center_lat = (data_bounds['lat_min'] + data_bounds['lat_max']) / 2
    center_lon = (data_bounds['lon_min'] + data_bounds['lon_max']) / 2
    
    lat_range = data_bounds['lat_max'] - data_bounds['lat_min'] + 2 * lat_margin
    lon_range = data_bounds['lon_max'] - data_bounds['lon_min'] + 2 * lon_margin
    zoom = calculate_initial_zoom(lat_range, lon_range)
    
    fig.update_layout(
        title="Clustered GPS Data Points in City of London",
        mapbox=dict(
            accesstoken=MAPBOX_ACCESS_TOKEN,
            style="streets",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
        ),
        autosize=True,
        height=800,
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    return fig.to_html(full_html=False)

@app.route('/')
def index():
    data = pd.read_csv('test_data.csv').to_numpy()
    max_points_per_cluster = 20
    max_distance = 0.002  # Adjust as needed (approx. 200 meters)

    clusters = constrained_kmeans(data, max_points_per_cluster, max_distance)
    
    data_bounds = {
        'lat_min': data[:, 0].min(),
        'lat_max': data[:, 0].max(),
        'lon_min': data[:, 1].min(),
        'lon_max': data[:, 1].max()
    }
    
    plot_html = generate_plot(clusters, data_bounds)
    
    return render_template('index.html', plot_html=plot_html)

if __name__ == '__main__':
    app.run(debug=True)