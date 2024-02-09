import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

image_path = image_path = r"C:\Users\lukas\OneDrive\Desktop\Bilder\zoey-deutch-65343-1653908496-w-893.jpg.jpg"
image = Image.open(image_path)

image_array = np.array(image)

height, width, channels = image_array.shape
X = image_array.reshape((height*width, channels))

n_clusters = 5

#KMeans clustering
k_means = KMeans(n_clusters=n_clusters)
k_means.fit(X)
k_means_labels = k_means.predict(X)
k_means_cluster_centers = k_means.cluster_centers_


#MiniBatchKmeans-Clusrering
mbk = MiniBatchKMeans(n_clusters=n_clusters)
mbk.fit(X)
mbk_labels = mbk.predict(X)
mbk_cluster_centers = mbk.cluster_centers_


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