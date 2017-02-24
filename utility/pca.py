from numpy import dot, linalg

class PCA:

    def __init__(self):
        pass

    def fit(self, x):
        sigma = dot(x.T, x) / len(x)
        return linalg.svd(sigma)

    def project(self, x, u, k):
        U_reduce = u[:, :k]
        return dot(x, U_reduce)
