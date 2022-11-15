import torch

class dgp_for_sa:
    def __init__(self, N, p):
        self.N = N
        self.p = p

    def expit(self,x):
        return torch.sigmoid(x)

    def generate_data(self):
        X = torch.rand(self.N, self.p)
        U = torch.randn(self.N, 1) * 0.1 + X[:, [0]] - X[:, [1]] ** 2
        kappa = 3 * torch.Tensor([(-1) ** i for i in range(self.p)]).reshape(self.p, 1)
        llambda = 4 * torch.Tensor([(-1) ** i for i in range(self.p)]).reshape(self.p, 1)
        gamma = 2
        delta = 2
        beta = 2
        pz = self.expit(X @ kappa + gamma * U)
        Z = torch.bernoulli(pz)
        py = self.expit(X @ llambda + beta * Z + delta * U)
        Y = torch.bernoulli(py)
        return X,Y,Z