import numpy as np
from tqdm import tqdm


class FCM():
    def __init__(
        self,
        data_shape: tuple,
        trainloop_cluster: int,
        trainloop_maxEpochs: int,
        trainloop_eps: float
    ) -> None:
        self.clusters = trainloop_cluster
        self.maxEpochs = trainloop_maxEpochs
        self.eps = trainloop_eps

        np.random.seed()
        self.W_init = np.random.rand(data_shape[0], self.clusters)
        self.W_init = self.__norm_Matrix(self.W_init)

    def train(
        self,
        X: np.array
    ) -> None:
        self.W = self.W_init.copy()

        self.error = []
        for n_c in tqdm(range(0, self.maxEpochs)):
            for j in range(0, self.clusters):
                c = np.expand_dims(
                    self.__cmeans_c_pr(self.W[:, j], X), axis=1
                ).transpose()
                if j == 0:
                    c_v = c
                else:
                    c_v = np.append(c_v, c, axis=0)

            for j in range(0, self.clusters):
                w = np.expand_dims(
                    self.__cmeans_w_pr(X, c_v[j, :], c_v),
                    axis=1
                )
                if j == 0:
                    W_new = w
                else:
                    W_new = np.append(W_new, w, axis=1)
            W_new = self.__norm_Matrix(W_new)

            self.error.append(np.linalg.norm(W_new - self.W))
            self.cmeans_c = c_v
            if (self.error[n_c] < self.eps):
                break

            self.W = W_new

    def invoke(
        self,
        X: np.array
    ) -> np.array:
        for j in range(0, self.clusters):
            w = np.expand_dims(
                self.__cmeans_w_pr(X, self.cmeans_c[j, :], self.cmeans_c),
                axis=1
            )
            if j == 0:
                W_new = w
            else:
                W_new = np.append(W_new, w, axis=1)
        return self.__norm_Matrix(W_new)

    def __norm_Matrix(
        self,
        W: np.array
    ) -> np.array:
        W[W > 1] = 1
        W[W < 0] = 0
        for k in range(0, W.shape[0]):
            W[k, :] = W[k, :] / np.sum(W, axis=1)[k]
        return W

    def __cmeans_w_pr(
        self,
        x: np.array,
        c_j: np.array,
        c_vec: np.array
    ) -> np.array:
        nom = np.linalg.norm(x - c_j, ord=1, axis=1)**(-2)
        den = 0
        for i in range(0, c_vec.shape[0]):
            den += np.sum(np.linalg.norm(x-c_vec[i, :], ord=1, axis=1)**(-2))
        w = nom / den
        return w

    def __cmeans_c_pr(
        self,
        w: np.array,
        x: np.array
    ) -> np.array:
        nom = np.sum(np.expand_dims(w, axis=1)**2 * x, axis=0)
        den = np.sum(np.expand_dims(w, axis=1)**2, axis=0)
        c = nom / den
        return c
