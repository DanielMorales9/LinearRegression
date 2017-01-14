import numpy as np


class FeatureAugmentation:

    @staticmethod
    def map(x, degree=2):
        """
            Maps the input features to quadratic features.
            Inputs X1, X2 must be the same size
            :param x: numpy array or sparse matrix of shape [n_samples, n_features]
                Dataset
            :param degree: int, optional
                maximum degree of the mapping, default is two.
            :return: x: numpy array or sparse matrix
                Mapped Feature
        """
        col = x.shape[1]
        out = np.ones((x.shape[0], 1))

        for n in range(col - 1):
            for m in range(n + 1, col):
                for i in range(1, degree):
                    for j in range(1, i + 1):
                        q = (x[:, n] ** i) * (x[:, m] ** j)
                        q = np.reshape(q, (q.shape[0], 1))
                        i -= 1
                        out = np.append(out, q, axis=1)

        for i in range(1, degree + 1):
            for n in range(col):
                q = x[:, n] ** i
                q = np.reshape(q, (q.shape[0], 1))
                out = np.append(out, q, axis=1)

        return out[:, 1:]

