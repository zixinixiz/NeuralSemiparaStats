import torch

class dgp_for_mnar:
    def __init__(self, N, p):
        self.N = N
        self.p = p

    def pi_oracle(self, y):
        return torch.exp(1 + y) / (1 + torch.exp(1 + y))

    def generate_data(self):

        X = torch.normal(0.5, 0.5, (self.N, self.p))
        X[:, 0] = 1
        beta_true = torch.FloatTensor([0.25, -0.5]).reshape(2, 1)
        E = torch.normal(0, 1, (self.N, 1))
        Y_nm = torch.mm(X, beta_true) + E
        R = torch.bernoulli(self.pi_oracle(Y_nm))
        Y = Y_nm * R

        return [X, Y, R]