import os
from clustering.kmeans import KMeans
import scipy.io as scio


path = os.path.dirname(os.path.abspath(__file__))
data = scio.loadmat(path+"/data/data.mat")

x = data['X']

k = 3
max_iter = 10

KMeans().fit(x, k, max_iter, plot_progress=True)
