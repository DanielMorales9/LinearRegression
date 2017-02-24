import matplotlib.pyplot as plt
from numpy import reshape
from clustering.kmeans import KMeans

im = plt.imread("data/bird_small.png")

im /= 255

x = reshape(im, (im.shape[0] * im.shape[1], 3))



km = KMeans()
idx, centroids = km.fit(x, k, max_iter)

idx = km.findClosestCentroids(x, centroids).astype(int)

x_recovered = centroids[idx, :]

X_recovered = reshape(x_recovered, (im.shape[0], im.shape[1], 3))

plt.subplot(1, 2, 1)
plt.imshow(im)
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(x_recovered)
plt.title('Compressed, with {} colors.'.format(k))
