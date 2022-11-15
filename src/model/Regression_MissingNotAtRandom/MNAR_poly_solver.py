import torch
import numpy as np
from scipy.optimize import fsolve
from sklearn.preprocessing import PolynomialFeatures
from src.model.Regression_MissingNotAtRandom.MNAR_utils import pi_star,dnorm,ddnorm

class PolySolver_for_mnar:

    def __init__(self,data_mnar,df,MC_SAMPLE_SIZE, TRAINING_SAMPLE_SIZE):
        self.data_mnar = data_mnar
        self.df = df
        self.pi = np.pi
        self.M = MC_SAMPLE_SIZE
        self.BATCHSIZE = TRAINING_SAMPLE_SIZE
        self.rnorm = torch.normal(0, 1, (self.M, 1)).numpy()


    def S_eff(self,beta):
        X,Y,R = self.data_mnar
        X,Y,R = X.numpy(),Y.numpy(),R.numpy()
        y_train = torch.linspace(int(min(Y)) - 1, int(max(Y)) + 1, self.BATCHSIZE).reshape(self.BATCHSIZE, 1)
        N = X.shape[0]
        poly = PolynomialFeatures(self.df, include_bias=False)

        beta = beta.reshape(-1, 1)
        int_denom = np.zeros([N, 1])
        intt = np.zeros([N, 2])
        int_poly = np.zeros([N, self.df])
        for i in range(N):
            int_denom[i] = 1 - np.mean(self.pi_star(self.rnorm + X[i, :] @ beta))
            intt[i, :] = np.mean(self.rnorm * self.pi_star(self.rnorm + X[i, :] @ beta)) * X[i, :]
            int_poly[i, :] = np.mean(poly.fit_transform(self.rnorm + X[i, :] @ beta) * self.pi_star(self.rnorm + X[i, :] @ beta),
                                     axis=0)
        def g0(y):
            return np.mean(
                self.ddnorm(y - X @ beta) * (-X[:, 0].reshape(-1, 1)) + intt[:, 0].reshape(-1, 1) / int_denom * self.dnorm(
                    y - X @ beta))

        def g1(y):
            return np.mean(
                self.ddnorm(y - X @ beta) * (-X[:, 1].reshape(-1, 1)) + intt[:, 1].reshape(-1, 1) / int_denom * self.dnorm(
                    y - X @ beta))

        def feature1(y):
            return np.mean(self.dnorm(y - X @ beta)) * poly.fit_transform(y.reshape(1, 1))  # 1 by df

        def feature2(y):
            return np.mean(int_poly / int_denom * self.dnorm(y - X @ beta), axis=0).reshape(1, -1)  # 1 by df

        X_reg = np.zeros([self.BATCHSIZE, self.df])
        y_reg0 = np.zeros([self.BATCHSIZE, 1])
        y_reg1 = np.zeros([self.BATCHSIZE, 1])
        for i in range(self.BATCHSIZE):
            yy = y_train[i]
            y_reg0[i] = g0(yy)
            y_reg1[i] = g1(yy)
            X_reg[i, :] = feature1(yy) + feature2(yy)

        alpha0 = np.linalg.inv(X_reg.T @ X_reg) @ (X_reg.T @ y_reg0)
        alpha1 = np.linalg.inv(X_reg.T @ X_reg) @ (X_reg.T @ y_reg1)

        def b0(y):
            return poly.fit_transform(y) @ alpha0

        def b1(y):
            return poly.fit_transform(y) @ alpha1

        int_b0 = np.zeros([N, 1])
        int_b1 = np.zeros([N, 1])
        for i in range(N):
            int_b0[i] = np.mean(b0(self.rnorm + X[i, :] @ beta) * self.pi_star(self.rnorm + X[i, :] @ beta))
            int_b1[i] = np.mean(b1(self.rnorm + X[i, :] @ beta) * self.pi_star(self.rnorm + X[i, :] @ beta))

        b_Y = np.concatenate((b0(Y), b1(Y)), axis=1)
        int_b = np.concatenate((int_b0, int_b1), axis=1)

        return np.mean(R / self.dnorm(Y - X @ beta) * self.ddnorm(Y - X @ beta) * (-X) - (1 - R) / int_denom * intt - R * b_Y + (
                1 - R) * int_b / int_denom, axis=0)

    def solver(self):
        return fsolve(self.S_eff(), [0, 0])