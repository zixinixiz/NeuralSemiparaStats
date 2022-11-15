import numpy as np
import torch
from scipy.optimize import fsolve
from sklearn.preprocessing import PolynomialFeatures

class PolySolver_for_sa:

    def __init__(self,data_sa, df=3, MC_SAMPLE_SIZE = 100, TRAINING_SAMPLE_SIZE = 5000):
        self.data_sa = data_sa
        self.df = df
        self.M = MC_SAMPLE_SIZE
        self.BATCHSIZE = TRAINING_SAMPLE_SIZE
        self.rnorm = torch.normal(0, 1, (self.M, 1)).numpy()

    def training_sample(self,sample_type):
        ones_M = torch.ones(self.M, 1)
        ones_N = torch.ones(self.N, 1)
        ones_B = torch.ones(self.BATCHSIZE, 1)
        if sample_type == 'random':
            return torch.hstack([torch.rand(self.B, 1) * 1.4 - 0.2 for _ in range(self.p + 1)])
        elif sample_type == 'partition_perm':
            partition = torch.linspace(-0.2, 1.2, self.B).reshape(self.B, 1)
            return torch.hstack([partition[index] for index in [torch.randperm(self.B) for _ in range(self.p + 1)]])
        elif sample_type == 'fixed_X':
            partition = torch.linspace(-0.2, 1.2, self.M).reshape(self.M, 1)
            X, Y, Z = self.data_sa
            return torch.hstack([torch.kron(ones_N, partition), torch.kron(X, ones_M)])

    def S_eff(self,beta):
        beta = np.array(beta).reshape(-1, 1)
        poly = PolynomialFeatures(degree=self.df, include_bias=False)

        X, Y, Z, kappa, gamma, llambda, delta = self.data_sa
        N, p = X.shape

        if self.v_sample == 'mesh':
            m = 100 + 1
            v = torch.linspace(0, 1, m).reshape(m, 1)
            h = 1 / (m - 1)
            W = (h * torch.diag(torch.hstack([0.5 * torch.ones(1), torch.ones(m - 2), 0.5 * torch.ones(1)])))
        elif self.v_sample == 'mc':
            m = 100 + 1
            v = torch.rand(m, 1) - 0.5

        data_train = self.training_sample(self.sample_type)
        data_train[:, [0]] = torch.rand(n, 1) - 0.5

        ones_4 = torch.ones(4, 1).numpy()
        ones_m = torch.ones(m, 1).numpy()
        ones_n = torch.ones(n, 1).numpy()
        ones_N = torch.ones(N, 1).numpy()
        ones_mn = torch.ones(m * n, 1).numpy()

        X, Y, Z, v = X.numpy(), Y.numpy(), Z.numpy(), v.numpy()
        data_train = data_train.numpy()

        def expit(x):
            return 1 / (1 + np.exp(-x))

        def fyz_vx(yzvx):
            y = yzvx[:, [0]].copy()
            z = yzvx[:, [1]].copy()
            v = yzvx[:, [2]].copy()
            x = yzvx[:, 3:(3 + p)].copy()

            pz = expit(np.dot(x, kappa) + gamma * v)
            py = expit(np.dot(x, llambda) + np.dot(z, beta) + delta * v)
            return (z * pz + (1 - z) * (1 - pz)) * (y * py + (1 - y) * (1 - py))

        def S(yzvx):
            y = yzvx[:, [0]].copy()
            z = yzvx[:, [1]].copy()
            v = yzvx[:, [2]].copy()
            x = yzvx[:, 3:(3 + p)].copy()

            py = expit(np.dot(x, llambda) + np.dot(z, beta) + delta * v)
            return (y - py) * z

        def solve_a(data_train, alpha):
            x = data_train[:, 1:(p + 1)]  # n by p
            yz = np.array([[y, z] for y in range(2) for z in range(2)])  # 4 by 2
            vx = np.hstack([np.kron(ones_n, v), np.kron(x, ones_m)])
            yzvx = np.hstack([np.kron(yz, ones_mn), np.kron(ones_4, vx)])
            p_yzvx = fyz_vx(yzvx)  # mn4 by 1
            s_yzvx = S(yzvx)  # mn4 by 1
            poly_vx = np.kron(ones_4, poly.fit_transform(vx))  # 4mn by 1

            if self.v_sample == 'mesh':
                nom_y = np.sum((s_yzvx * p_yzvx).reshape(4, n, m)@W, axis=2)
                nom_x = [np.sum((poly_vx[:, [j]] * p_yzvx).reshape(4, n, m)@W, axis=2) for j in range(poly_vx.shape[1])]
                denom = np.sum(p_yzvx.reshape(4, n, m)@W, axis=2)
            elif self.v_sample == 'mc':
                nom_y = np.mean((s_yzvx * p_yzvx).reshape(4, n, m), axis=2)
                nom_x = [np.mean((poly_vx[:, [j]] * p_yzvx).reshape(4, n, m), axis=2) for j in range(poly_vx.shape[1])]
                denom = np.mean(p_yzvx.reshape(4, n, m), axis=2)  # 4 by n

            # sum over yz
            yzux = np.hstack([np.kron(yz, ones_n), np.kron(ones_4, data_train)])  # 4n by 1
            p_yzux = fyz_vx(yzux).reshape(4, n)  # 4 by n

            poly_ux = poly.fit_transform(data_train)
            X_reg = np.hstack(
                [np.sum(nom_x[j] / denom * p_yzux, axis=0).reshape(n, 1) for j in range(len(nom_x))]) + alpha * poly_ux
            y_reg = np.sum(nom_y / denom * p_yzux, axis=0).reshape(n, 1)
            d = X_reg.shape[1]
            return np.linalg.inv((X_reg.T @ X_reg)) @ ((X_reg.T @ y_reg))

        coef = solve_a(data_train=data_train, alpha=alpha)

        def a(ux):
            return poly.fit_transform(ux) @ coef

        vX = np.hstack([np.kron(ones_N, v), np.kron(X, ones_m)])  # mN by 3
        a_vX = a(vX)  # mN by 1
        YZvX = np.hstack([np.kron(Y, ones_m), np.kron(Z, ones_m), vX])  # mN by 5

        s_YZvX = S(YZvX)  # mN by 1
        p_YZvX = fyz_vx(YZvX)  # mN by 1

        if self.v_sample == 'mesh':
            nom = np.sum(((s_YZvX - a_vX) * p_YZvX).reshape(N, m)@W, axis=1).reshape(N, 1)
            denom = np.sum(p_YZvX.reshape(N, m)@W, axis=1).reshape(N, 1)
        elif self.v_sample == 'mc':
            nom = np.mean(((s_YZvX - a_vX) * p_YZvX).reshape(N, m), axis=1).reshape(N, 1)  # N
            denom = np.mean(p_YZvX.reshape(N, m), axis=1).reshape(N, 1)

        return np.mean(nom / denom)

    def solver(self):
        return fsolve(self.S_eff(), [0])