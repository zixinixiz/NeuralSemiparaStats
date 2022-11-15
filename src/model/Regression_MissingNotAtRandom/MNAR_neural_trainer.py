import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from src.model.Regression_MissingNotAtRandom.MNAR_utils import pi_star,dnorm,ddnorm


class NeuralNetsTrainer_for_mnar:
    def __init__(self, data_mnar, MC_SAMPLE_SIZE = 1000, TRAINING_SAMPLE_SIZE = 5000, EPOCH = 20000,
                 BATCHSIZE = 5000, width = 20,depth = 0,nn_ac_fun = 'Tanh', device = 'cuda:0'):

        self.data_mnar = data_mnar
        self.M = MC_SAMPLE_SIZE
        self.B = TRAINING_SAMPLE_SIZE
        self.EPOCH = EPOCH
        self.BATCHSIZE = BATCHSIZE
        self.w = width
        self.d = depth
        if nn_ac_fun == 'Tanh':
            self.nn_ac_fun = nn.Tanh()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.pi = torch.FloatTensor([np.pi]).to(device)


    def model(self):
        p = self.data_mnar[0].shape[1]
        beta = nn.Linear(p, 1, bias=False)

        b0 = nn.Sequential(
            nn.Linear(1, self.w),
            self.nn_ac_fun,
            *[nn.Linear(self.w , self.w ) if i % 2 == 0 else nn.ReLU() for i in range(self.d * 2)],
            nn.Linear(self.w, 1)
        )

        b1 = nn.Sequential(
            nn.Linear(1, self.w),
            self.nn_ac_fun,
            *[nn.Linear(self.w, self.w) if i % 2 == 0 else nn.ReLU() for i in range(self.d * 2)],
            nn.Linear(self.w, 1)
        )
        return beta, b0, b1

    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def train(self):

        X,Y,R = self.data_mnar
        X,Y,R = X.to(self.device), Y.to(self.device), R.to(self.device)

        beta, b0, b1 = self.model()
        beta, b0, b1 = beta.to(self.device), b0.to(self.device), b1.to(self.device)

        b0.apply(self.weight_init())
        b1.apply(self.weight_init())
        beta.apply(self.weight_init())

        y_train0 = torch.linspace(int(min(Y)) - 0.5, int(max(Y)) + 0.5, self.B).reshape(self.B, 1)
        train_data = Data.TensorDataset(y_train0, torch.zeros(self.B, 1))
        train_loader = Data.DataLoader(dataset=train_data, batch_size=self.BATCHSIZE, shuffle=True)

        # set parameters and optimizer
        Parameters_b = [{"params": b0.parameters()}, {"params": b1.parameters()}]  # ,{"params":beta.parameters()}]
        optimizer_b = torch.optim.Adam(Parameters_b, lr=1e-3)
        Parameters_beta = beta.parameters()
        optimizer_beta = torch.optim.Adam(Parameters_beta, lr=1e-2)

        ones_M = torch.ones(self.M, 1).to(self.device)
        ones_N = torch.ones(self.N, 1).to(self.device)
        ones_B = torch.ones(self.BATCHSIZE, 1).to(self.device)

        writer = SummaryWriter()
        for epoch in range(self.EPOCH):
            for step, (y_train, _) in enumerate(train_loader):
                y_train = y_train.to(self.device)
                rnorm = torch.normal(0, 1, (self.M, 1)).to(self.device)

                aa = torch.kron(ones_N, rnorm) + torch.kron(beta(X).data.to(self.device), ones_M)
                int_denom = 1 - torch.mean(pi_star(aa.reshape(self.N, self.M)), dim=1).reshape(-1, 1)
                bb = torch.mean((torch.kron(ones_N, rnorm) * pi_star(aa)).reshape(self.N, self.M), dim=1).reshape(-1, 1)
                int0 = bb * X[:, 0].reshape(-1, 1)
                int1 = bb * X[:, 1].reshape(-1, 1)
                int_b0 = torch.mean((b0(aa) * pi_star(aa)).reshape(self.N, self.M), dim=1).reshape(-1, 1)
                int_b1 = torch.mean((b1(aa) * pi_star(aa)).reshape(self.N, self.M), dim=1).reshape(-1, 1)

                kron_betaX = torch.kron(ones_B, beta(X).data.to(self.device))
                kron_int_denom = torch.kron(ones_B, int_denom)

                def h(y):
                    return torch.mean(dnorm(y - kron_betaX).reshape(self.BATCHSIZE, self.N), dim=1).reshape(-1, 1)

                def f0(y):
                    return torch.mean(
                        (torch.kron(ones_B, int_b0) / kron_int_denom * dnorm(y - kron_betaX)).reshape(self.BATCHSIZE, self.N),
                        dim=1).reshape(-1, 1)

                def f1(y):
                    return torch.mean(
                        (torch.kron(ones_B, int_b1) / kron_int_denom * dnorm(y - kron_betaX)).reshape(self.BATCHSIZE, self.N),
                        dim=1).reshape(-1, 1)

                def g0(y):
                    return torch.mean(
                        (ddnorm(y - kron_betaX) * torch.kron(ones_B, -X[:, 0].reshape(self.N, 1)) + torch.kron(ones_B,
                                                                                                          int0) / kron_int_denom * dnorm(
                            y - kron_betaX)).reshape(self.BATCHSIZE, self.N)
                        , dim=1).reshape(-1, 1)

                def g1(y):
                    return torch.mean(
                        (ddnorm(y - kron_betaX) * torch.kron(ones_B, -X[:, 1].reshape(self.N, 1)) + torch.kron(ones_B,
                                                                                                          int1) / kron_int_denom * dnorm(
                            y - kron_betaX)).reshape(self.BATCHSIZE, self.N)
                        , dim=1).reshape(-1, 1)

                y = torch.kron(y_train, ones_N)
                res0 = torch.mean((b0(y_train) * h(y) + f0(y) - g0(y)) ** 2)
                res1 = torch.mean((b1(y_train) * h(y) + f1(y) - g1(y)) ** 2)
                loss_b = res0 + res1

                optimizer_b.zero_grad()
                loss_b.backward()
                optimizer_b.step()

            if epoch % self.alternating_freq == 0:
                rnorm = torch.normal(0, 1, (self.M, 1)).to(self.device)

                aa = torch.kron(ones_N, rnorm) + torch.kron(beta(X), ones_M)
                int_denom = 1 - torch.mean(self.pi_star(aa.reshape(self.N, self.M).t()), dim=0).reshape(-1, 1)
                bb = torch.mean((torch.kron(ones_N, rnorm) * self.pi_star(aa)).reshape(self.N, self.M), dim=1).reshape(-1, 1)
                int0 = bb * X[:, 0].reshape(-1, 1)
                int1 = bb * X[:, 1].reshape(-1, 1)
                int_b0 = torch.mean((b0(aa).data.to(self.device) * pi_star(aa)).reshape(self.N, self.M), dim=1).reshape(-1, 1)
                int_b1 = torch.mean((b1(aa).data.to(self.device) * pi_star(aa)).reshape(self.N, self.M), dim=1).reshape(-1, 1)

                S_eff0 = torch.mean((R / dnorm(Y - beta(X)) * ddnorm(Y - beta(X)) * (-X[:, 0]).reshape(-1, 1) - (
                        1 - R) / int_denom * int0 - R * b0(Y).data.to(self.device) + (1 - R) / int_denom * int_b0))
                S_eff1 = torch.mean((R / dnorm(Y - beta(X)) * ddnorm(Y - beta(X)) * (-X[:, 1]).reshape(-1, 1) - (
                        1 - R) / int_denom * int1 - R * b1(Y).data.to(self.device) + (1 - R) / int_denom * int_b1))
                loss_beta = S_eff0 ** 2 + S_eff1 ** 2

                optimizer_beta.zero_grad()
                loss_beta.backward()
                optimizer_beta.step()

            if epoch % 50 == 0:
                writer.add_scalar('loss_nn', loss_b, global_step=epoch)
                writer.add_scalar('loss_beta', loss_beta, global_step=epoch)
                writer.add_scalar('beta0', beta.weight[0][0], global_step=epoch)
                writer.add_scalar('beta1', beta.weight[0][1], global_step=epoch)

        return beta.weight.item()