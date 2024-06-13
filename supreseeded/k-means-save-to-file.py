import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

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

def plot_clusters(clusters, filename='clusters.png'):
    plt.figure(figsize=(10, 8))
    colors = plt.colormaps.get_cmap('tab20')
    num_colors = colors.N

    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        color = colors(i % num_colors)
        plt.scatter(cluster[:, 0], cluster[:, 1], s=50, color=color, label=f'Cluster {i+1}')

    plt.title('Clustered Data Points')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.savefig(filename)
    plt.close()

# Example usage
data = np.random.rand(100, 2)  # Replace with your actual data
max_points_per_cluster = 20
max_distance = 0.2  # Adjust as needed

clusters = constrained_kmeans(data, max_points_per_cluster, max_distance)

# Plot the clusters and save to a file
plot_clusters(clusters, 'clusters.png')

print("Plot saved as 'clusters.png'")
