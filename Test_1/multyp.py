import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
import time

image_path = image_path = r"C:/Users/lukas/OneDrive/Desktop/Bilder/1134871.jpg"
image = Image.open(image_path)

image_array = np.array(image)

height, width, channels = image_array.shape
X = image_array.reshape((height*width, channels))

n_clusters = 5

#KMeans clustering
start_time = time.time()
k_means = KMeans(n_clusters=n_clusters)
k_means.fit(X)
k_means_labels = k_means.predict(X)
k_means_cluster_centers = k_means.cluster_centers_
k_means_time = time.time() - start_time

#MiniBatchKmeans-Clusrering
start_time = time.time()
mbk = MiniBatchKMeans(n_clusters=n_clusters)
mbk.fit(X)
mbk_labels = mbk.predict(X)
mbk_cluster_centers = mbk.cluster_centers_
mbk_time = time.time() - start_time


order = pairwise_distances_argmin(k_means_cluster_centers, mbk_cluster_centers)

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


print('KMeans Cluster Centers')
print(k_means_cluster_centers)

print('MiniBatchKMeans Cluster Centers:')
print(mbk_cluster_centers[order[mbk_labels]])

k_means_colors = k_means_cluster_centers.astype(np.uint8)
mbk_colors = mbk_cluster_centers[order[mbk_labels]].astype(np.uint8)

print('KMEans Cluster Colors:')
print(k_means_colors)
print('MiniBatchKMEans Cluster Colors:')
print(mbk_colors)

print("Zeit für KMeans-Clustering:", k_means_time, "sec")
print("Zeit für MiniBatchKMeans-Clustering:", mbk_time, "secn")