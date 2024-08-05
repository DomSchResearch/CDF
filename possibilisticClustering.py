import numpy as np
from tqdm import tqdm


class PossC():
    def __init__(
        self,
        data_shape: tuple,
        trainloop_cluster: int,
        trainloop_maxEpochs: int,
        trainloop_eps: float,
        algorithm_beta: float,
        algorithm_eta: float,
    ) -> None:
        self.clusters = trainloop_cluster
        self.maxEpochs = trainloop_maxEpochs
        self.eps = trainloop_eps
        self.beta = algorithm_beta
        self.eta = algorithm_eta

        np.random.seed()
        self.W_init = np.random.rand(data_shape[0], self.clusters)
        self.W_init = self.__norm_Matrix(self.W_init)
        self.c_vec_init = np.random.rand(self.clusters, data_shape[1])
        self.beta_vec = np.ones((data_shape[1], ))

    def train(
        self,
        X: np.array
    ) -> None:
        self.W = self.W_init.copy()
        c_vec = self.c_vec_init

        self.error = []
        for n_poss in tqdm(range(0, self.maxEpochs)):
            for k in range(0, X.shape[0]):
                for j in range(0, self.clusters):
                    c = self.__possclust_c(
                        X[k, :],
                        c_vec[j, :],
                        self.eta,
                        self.beta,
                        self.beta_vec,
                        self.W[k, j]
                    )
                    c = np.expand_dims(c, axis=1).transpose()
                    if j == 0:
                        c_v = c
                    else:
                        c_v = np.append(c_v, c, axis=0)
                c_vec = c_v

                for j in range(0, self.clusters):
                    mü = self.__possclust_mü(
                        X[0:(k+1), :],
                        c_vec[j, :],
                        self.W[0:(k+1), j],
                        self.beta_vec,
                        self.beta
                    )
                    mü = np.expand_dims(np.array(mü), axis=0)
                    if j == 0:
                        mü_v = mü
                    else:
                        mü_v = np.append(mü_v, mü, axis=0)

                for j in range(0, self.clusters):
                    w = self.__possclust_w(
                        X[k, :],
                        c_v[j, :],
                        self.beta,
                        self.beta_vec,
                        mü_v[j]
                    )
                    w = np.expand_dims(np.array(w), axis=(0, 1))
                    if j == 0:
                        W_new_j = w
                    else:
                        W_new_j = np.append(W_new_j, w, axis=1)

                if k == 0:
                    W_new = W_new_j
                else:
                    W_new = np.append(W_new, W_new_j, axis=0)

            W_new = self.__norm_Matrix(W_new)

            self.error.append(np.linalg.norm(W_new - self.W))
            self.possclust_cv = c_v
            self.poss_mü = mü_v
            if (self.error[n_poss] <= self.eps):
                break

            self.W = W_new

    def invoke(
        self,
        X: np.array
    ) -> np.array:
        for k in range(0, X.shape[0]):
            for j in range(0, self.clusters):
                w = self.__possclust_w(
                    X[k, :],
                    self.possclust_cv[j, :],
                    self.beta,
                    self.beta_vec,
                    self.poss_mü[j]
                )
                w = np.expand_dims(np.array(w), axis=(0, 1))
                if j == 0:
                    W_new_j = w
                else:
                    W_new_j = np.append(W_new_j, w, axis=1)

            if k == 0:
                W_new = W_new_j
            else:
                W_new = np.append(W_new, W_new_j, axis=0)

        return self.__norm_Matrix(W_new)

    def __possclust_w(
        self,
        x: np.array,
        c_j: np.array,
        beta: float,
        beta_vec: np.array,
        mü: float
    ) -> np.array:
        return (1 + (self.__Dr1(beta_vec, x, c_j)/mü)**(1/(beta-1)))**(-1)

    def __possclust_c(
        self,
        x: np.array,
        c_vec: np.array,
        eta: float,
        beta: float,
        beta_vec: np.array,
        w: np.array
    ) -> np.array:
        term = eta*w**(beta)*np.tanh((x-c_vec)/beta_vec)
        c_new = c_vec + term
        return c_new

    def __possclust_mü(
        self,
        x: np.array,
        c_j: np.array,
        w: np.array,
        beta_vec: np.array,
        beta: float
    ) -> float:
        nom = np.sum(
            (beta_vec * np.log(np.cosh(((x - c_j) / beta_vec)))),
            axis=1
        )
        nom = np.sum(w**beta * nom)
        denom = np.sum(w**beta)
        mü = nom / denom
        return mü

    def __Dr1(
        self,
        beta_v: np.array,
        x: np.array,
        c: np.array
    ) -> np.array:
        return np.sum(beta_v*np.log(np.cosh((x-c)/beta_v)))

    def __norm_Matrix(
        self,
        W: np.array
    ) -> np.array:
        W[W > 1] = 1
        W[W < 0] = 0
        for k in range(0, W.shape[0]):
            W[k, :] = W[k, :] / np.sum(W, axis=1)[k]
        return W
