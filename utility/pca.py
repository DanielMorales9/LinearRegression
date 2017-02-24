from numpy import dot, linalg

class PCA:

    def __init__(self):
        pass

    def fit(self, x):
        sigma = dot(x.T, x) / float(len(x))
        u, s, _ = linalg.svd(sigma, full_matrices=0)
        return u, s

    def project(self, x, u, k):
        U_reduce = u[:, :k]
        return dot(x, U_reduce)

    def recover(self, z, u, k):
        U_reduce = u[:, :k]
        return dot(z, U_reduce.T);
