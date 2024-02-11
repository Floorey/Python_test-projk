import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import time

# Laden des Bildes
image_path = r"C:\Users\lukas\OneDrive\Desktop\Bilder\1134871.jpg"
image = Image.open(image_path)

# Merkmale extrahieren
image_array = np.array(image)
features = []
for channel in range(3):  # Iteriere über die Farbkanäle (Rot, Grün, Blau)
    channel_data = image_array[:, :, channel]
    features.append(channel_data.flatten())

X = np.vstack(features).T  # Merkmale zusammenführen

# DBSCAN-Parameter
eps = 10
min_samples = 5

# DBSCAN-Clustering
start_time = time.time()
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(X)
dbscan_time = time.time() - start_time

# Anzahl der Cluster zählen
unique_labels = set(dbscan_labels)
n_clusters_dbscan = len(unique_labels) - (1 if -1 in dbscan_labels else 0)
print("Anzahl der Cluster (DBSCAN):", n_clusters_dbscan)
print("Zeit für DBSCAN-Clustering:", dbscan_time, "Sekunden")

# Clusterergebnis auf die Form des Originalbildes zurückprojizieren
clustered_image = dbscan_labels.reshape((image_array.shape[0], image_array.shape[1]))

# Originalbild und Clusteringergebnis anzeigen
plt.figure(figsize=(12, 6))

# Originalbild anzeigen
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Originalbild")

# DBSCAN-Clusteringergebnis visualisieren
plt.subplot(1, 2, 2)
plt.imshow(clustered_image, cmap='viridis')
plt.title("DBSCAN Clustering")

plt.show()
