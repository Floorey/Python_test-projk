try:
    from scipy.datasets import face
except ImportError:
    from scipy.misc import face
import matplotlib.pyplot as plt    
from sklearn.preprocessing import KBinsDiscretizer

    
racoon_face = face(gray=True)

print(f'The dimension of the image is {racoon_face}')
print(f'The data used to encode the image is pf type {racoon_face.dtype}')
print(f'The number of bytes taken in RAM is {racoon_face.nbytes}')

fig, ax = plt.subplots(ncols=2, figsize=(12, 4))

ax[0].imshow(racoon_face, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Rendering of the image')
ax[1].hist(racoon_face.ravel(), bins=256)
ax[1].set_xlabel('Pixel value')
ax[1].set_ylabel('Count of pixels')
ax[1].set_title('Disribution of the pixels values')
_ = fig.suptitle('Original image of a racoon face')   

plt.show()

n_bins = 8

encoder = KBinsDiscretizer(
    n_bins=n_bins, encode='ordinal',
    strategy='uniform',
    random_state=0,
    subsample=200_000,
)

compressed_racoon_uniform = encoder.fit_transform(racoon_face.reshape(-1,1)).reshape(
    racoon_face.shape
)
fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
ax[0].imshow(compressed_racoon_uniform, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Rendering of the image')
ax[1].hist(compressed_racoon_uniform.ravel(), bins=256)
ax[1].set_xlabel('Pixel value')
ax[1].set_ylabel('Count of pixels')
ax[1].set_title('Sub-sampled distribution of the pixels values')
_ = fig.suptitle('Racoon face compressed using 3 bits and uniform strategy')

plt.show()

bin_edges = encoder.bin_edges_[0]
bin_center = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2

_, ax = plt.subplots()
ax.hist(racoon_face.ravel(), bins=256)
color = 'tab:orange'
for center in bin_center:
    ax.axvline(center, color=color)
    ax.text(center - 10, ax.get_ybound()[1]+100, f'{center:.1f}', color=color)
plt.show()
