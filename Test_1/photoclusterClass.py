from time import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle


class ImageQuantizer:
    def __init__(self, n_colors=64, image_path=None):
        self.n_colors = n_colors
        self.image_path = image_path
        self.image = None
        self.image_array = None
        self.reconstructed_image = None

    def load_image(self):
        self.image = plt.imread(self.image_path)
        self.image = np.array(self.image, dtype=np.float64) / 255
        h, w, d = self.image.shape
        assert d == 3
        self.image_array = np.reshape(self.image, (w * h, d))
        return w, h


    def fit_model(self, w, h):
        print('Fitting model on a small sub-sample of the data')
        t0 = time()
        image_array_sample = shuffle(self.image_array, random_state=0)[:1000]
        kmeans = KMeans(n_clusters=self.n_colors, random_state=0).fit(image_array_sample)
        self.reconstructed_image = self._recreate_image(kmeans.cluster_centers_, kmeans.labels_, w, h)
        print(f'done in {time() - t0:0.3f}s.')


    def calculate_mse(self):
        mse = np.mean((self.image - self.reconstructed_image) ** 2)
        return mse

    def show_images(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(self.image)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title(f'Quantized Image ({self.n_colors} colors)')
        plt.imshow(self.reconstructed_image)
        plt.axis('off')

        plt.show()

    def _recreate_image(self, codebook, labels, w, h):
        return codebook[labels].reshape(h, w, -1)



quantizer = ImageQuantizer(n_colors=64, image_path="C:/Users/lukas/OneDrive/Desktop/Bilder/pleiades-1.jpg")
w, h = quantizer.load_image()
quantizer.fit_model(w, h)
mse = quantizer.calculate_mse()
print(f'Mean Squared Error: {mse}')
quantizer.show_images()
