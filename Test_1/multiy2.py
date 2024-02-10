import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import silhouette_score
import time

# Laden des Bildes
image_path = r"C:\Users\lukas\OneDrive\Desktop\Bilder\zoey-deutch-65343-1653908496-w-893.jpg.jpg"
image = Image.open(image_path)

# Umwandeln des Bildes in ein NumPy-Array
image_array = np.array(image)

# Umformen des NumPy-Arrays, um die Pixel als Features zu haben
height, width, channels = image_array.shape
X = image_array.reshape((height * width, channels))

# Verschiedene Anzahlen von Clustern ausprobieren
cluster_range = range(2, 11)
k_means_times = []
mbk_times = []
k_means_silhouette_scores = []
mbk_silhouette_scores = []

for n_clusters in cluster_range:
    # KMeans-Clustering
    start_time = time.time()
    k_means = KMeans(n_clusters=n_clusters)
    k_means.fit(X)
    k_means_time = time.time() - start_time
    k_means_times.append(k_means_time)
    k_means_labels = k_means.predict(X)
    k_means_silhouette_scores.append(silhouette_score(X, k_means_labels))

    # MiniBatchKMeans-Clustering
    start_time = time.time()
    mbk = MiniBatchKMeans(n_clusters=n_clusters)
    mbk.fit(X)
    mbk_time = time.time() - start_time
    mbk_times.append(mbk_time)
    mbk_labels = mbk.predict(X)
    mbk_silhouette_scores.append(silhouette_score(X, mbk_labels))

# Ergebnisse plotten
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, k_means_times, '-o', label='KMeans Zeit')
plt.plot(cluster_range, mbk_times, '-o', label='MiniBatchKMeans Zeit')
plt.plot(cluster_range, k_means_silhouette_scores, '-o', label='KMeans Silhouette Score')
plt.plot(cluster_range, mbk_silhouette_scores, '-o', label='MiniBatchKMeans Silhouette Score')
plt.xlabel('Anzahl der Cluster')
plt.ylabel('Wert')
plt.title('Auswirkung der Anzahl der Cluster auf die Zeit und Silhouette Score')
plt.legend()
plt.show()

# Wähle die Anzahl der Cluster für die detaillierte Analyse (z.B. die letzte)
n_clusters = cluster_range[-1]

# KMeans-Clustering
k_means = KMeans(n_clusters=n_clusters)
k_means.fit(X)
k_means_labels = k_means.predict(X)
k_means_cluster_centers = k_means.cluster_centers_

# MiniBatchKMeans-Clustering
mbk = MiniBatchKMeans(n_clusters=n_clusters)
mbk.fit(X)
mbk_labels = mbk.predict(X)
mbk_cluster_centers = mbk.cluster_centers_

# Berechnen der Zuordnung der Clusterzentren zwischen KMeans und MiniBatchKMeans
order = pairwise_distances_argmin(k_means_cluster_centers, mbk_cluster_centers)

# Visualisierung der Ergebnisse und Zeit für jedes Clustering
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Originalbild
axes[0].imshow(image)
axes[0].set_title("Original")

# KMeans-Clustering
axes[1].imshow(k_means_cluster_centers[k_means_labels].reshape(height, width, channels).astype(np.uint8))
axes[1].set_title("KMeans Clustered")

# MiniBatchKMeans-Clustering
axes[2].imshow(mbk_cluster_centers[order[mbk_labels]].reshape(height, width, channels).astype(np.uint8))
axes[2].set_title("MiniBatchKMeans Clustered")

plt.show()

# Ausgabe der Clusterzentren und Farben für KMeans und MiniBatchKMeans
print('KMeans Cluster Centers:')
print(k_means_cluster_centers)

print('MiniBatchKMeans Cluster Centers:')
print(mbk_cluster_centers[order[mbk_labels]])

k_means_colors = k_means_cluster_centers.astype(np.uint8)
mbk_colors = mbk_cluster_centers[order[mbk_labels]].astype(np.uint8)

print('KMeans Cluster Colors:')
print(k_means_colors)
print('MiniBatchKMeans Cluster Colors:')
print(mbk_colors)

# Ausgabe der Anzahl der Cluster und der Zeit für jedes Clustering
print("Anzahl der Cluster (KMeans):", n_clusters)
print("Anzahl der Cluster (MiniBatchKMeans):", n_clusters)
