from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt

class ImageSorter:
    def __init__(self, max_num_cluster=10, batch_size=100):
        self.max_num_cluster = max_num_cluster
        self.batch_size = batch_size
        self.kmeans = MiniBatchKMeans(n_clusters=1, batch_size=self.batch_size)

    def find_optimal_num_cluster(self, img_array):
        inertia_values = []
        for num_cluster in range(1, self.max_num_cluster + 1):
            kmeans = MiniBatchKMeans(n_clusters=num_cluster, batch_size=self.batch_size, verbose=True)
            kmeans.fit(img_array)
            inertia_values.append(kmeans.inertia_)
            
        optimal_num_cluster = 1
        
        min_dist = float('inf')
        for i in range(1, len(inertia_values)):
            dist = abs(inertia_values[i] - inertia_values[i-1])
            if dist < min_dist:
                min_dist = dist
                optimal_num_cluster = i + 1
        return optimal_num_cluster

    def cluster_image(self, image_path):
        img = Image.open(image_path)
        img_array = np.array(img)
        reshaped_img_array = img_array.reshape(-1, 3)
        self.kmeans = MiniBatchKMeans(n_clusters=self.find_optimal_num_cluster(reshaped_img_array), batch_size=self.batch_size, verbose=True)
        self.kmeans.fit(reshaped_img_array)
        self.labels = self.kmeans.predict(reshaped_img_array)  # Speichern der Labels

    def visualize_clusters(self, img_array):
        cluster_centers = self.kmeans.cluster_centers_

        print("Cluster Centers:", cluster_centers)

        # Labels für die Pixel abrufen
        labels = self.kmeans.predict(img_array)

        # Neues Bild erstellen, das die Clusterzentren enthält
        cluster_img = np.zeros_like(img_array)

        # Jeden Pixel entsprechend seinem Cluster einfärben
        for label, color in zip(np.unique(labels), cluster_centers):
            cluster_img[labels == label] = color

        # Sortieren der Pixel
        sorted_indices = np.argsort(labels)
        sorted_cluster_img = cluster_img[sorted_indices]

        # Bild zurück in das ursprüngliche Format bringen
        sorted_cluster_img = sorted_cluster_img.reshape(img_array.shape)

        # Plot der Cluster als farbige Balken ohne Beschriftung
        plt.imshow(sorted_cluster_img)
        plt.axis('off')

        # Anpassung der Balkenbreite
        num_clusters = len(cluster_centers)
        bar_width = 0.5 / num_clusters  # Breite der Balken, z.B. 0.8 für 80% der Breite zwischen den Clusterzentren
        for i in range(num_clusters):
            color = cluster_centers[i] / 255.0  # Konvertiere RGB-Werte von 0 bis 255 auf 0 bis 1
            plt.bar(i, 1, color=color, width=bar_width, edgecolor='none')

        plt.show()

if __name__ == "__main__":
    img_path = "C:/Users/lukas/OneDrive/Desktop/Bilder/Seattle_4.jpg"
    img = Image.open(img_path)
    img_array = np.array(img)
    width, height = img.size
    print('Pixel Count:', width * height)
    reshaped_img_array = img_array.reshape(-1, 3)

    sorter = ImageSorter()
    sorter.cluster_image(img_path)
    sorter.visualize_clusters(reshaped_img_array)
