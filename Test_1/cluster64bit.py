from time import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle



def calculate_mse(original_image, quantized_image):
    mse = np.mean((original_image - quantized_image) ** 2)
    return mse

n_colors = 64


china = plt.imread("C:/Users/lukas/OneDrive/Desktop/Bilder/pleiades-1.jpg")


china = np.array(china, dtype=np.float64) / 255


w, h, d = original_shape = tuple(china.shape)
assert d == 3


image_array = np.reshape(china, (w * h, d))

print('Fitting model on a small sub-sample of the data')
t0 = time()

image_array_sample = shuffle(image_array, random_state=0)[:1000]

kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print(f'done in {time() - t0:0.3f}s.')

print('Predicting color indices on the full image (k-means)')
t0 = time()
labels = kmeans.predict(image_array)
print(f'done in {time() - t0:0.3f}s.')




def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    return codebook[labels].reshape(w, h, -1)

reconstructed_image = recreate_image(kmeans.cluster_centers_, labels, w, h)

mse = calculate_mse(china, reconstructed_image)
print(f'MSE:{mse}')

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(china)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f'Quantized Image ({n_colors} colors)')
plt.imshow(reconstructed_image)
plt.axis('off')

plt.show()
