from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Vorbereite die Adressdaten für das Clustering
addresses = [user['address'] for user in user_list]

# Verwende einfache Merkmale für das Clustering (z.B. Länge der Adresse)
features = [[len(address)] for address in addresses]

# Standardisiere die Daten
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Wähle die Anzahl der Cluster
num_clusters = 3

# Führe das Clustering durch
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Füge die Clusterinformationen zu den Benutzerdaten hinzu
for i, user in enumerate(user_list):
    user['cluster'] = clusters[i]

# Visualisiere die Cluster
plt.scatter(features_scaled[:, 0], [0] * len(features_scaled), c=clusters, cmap='viridis', alpha=0.5)
plt.title('Address Clustering')
plt.xlabel('Scaled Address Length')
plt.show()
