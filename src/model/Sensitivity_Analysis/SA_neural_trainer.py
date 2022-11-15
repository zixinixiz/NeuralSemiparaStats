import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

class NeuralNetsTrainer_for_sa:
    def __init__(self, data_sa, v_sample = 'random', alpha = 0.01, alter_freq = 1,
                 MC_SAMPLE_SIZE = 1000, TRAINING_SAMPLE_SIZE = 5000, EPOCH = 20000,
                 BATCHSIZE = 5000, width = 20,depth = 0,nn_ac_fun = 'ReLU', device = 'cuda:0'):

        self.data_sa = data_sa
        self.alter_freq = alter_freq
        self.M = MC_SAMPLE_SIZE
        self.B = TRAINING_SAMPLE_SIZE
        self.n = BATCHSIZE
        self.EPOCH = EPOCH
        self.BATCHSIZE = BATCHSIZE
        self.w = width
        self.d = depth
        self.alpha = alpha
        if nn_ac_fun == 'Tanh':
            self.nn_ac_fun = nn.Tanh()
        elif nn_ac_fun == 'ReLU':
            self.nn_ac_fun = nn.ReLU()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.v_sample = v_sample

    def expit(self,x):
        return torch.sigmoid(x)

    def model(self):
        p = self.data_mnar[0][1]
        beta_net = nn.Linear(1, 1, bias=False)
        a = nn.Sequential(
            nn.Linear(1 + p, self.w * (1 + p)),
            self.nn_ac_fun,
            *[nn.Linear(self.w * (1 + p), self.w * (1 + p)) if i % 2 == 0 else self.nn_ac_fun for i in range(self.d * 2)],
            nn.Linear(self.w * (1 + p), 1)
        )
        return a, beta_net

    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def weight_init_beta(w):
        if isinstance(w, nn.Linear):
            nn.init.constant_(w.weight, 0)

    def adjust_learning_rate(optimizer, epoch, lr):
        if epoch <= 100:
            lr = 1e-2
        elif epoch <= 200:
            lr = 1e-2
        else:
            lr = 1e-2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def training_sample(self,sample_type):
        ones_M = torch.ones(self.M, 1).to(self.device)
        ones_N = torch.ones(self.N, 1).to(self.device)
        ones_B = torch.ones(self.BATCHSIZE, 1).to(self.device)
        if sample_type == 'random':
            return torch.hstack([torch.rand(self.B, 1) * 1.4 - 0.2 for _ in range(self.p + 1)])
        elif sample_type == 'partition_perm':
            partition = torch.linspace(-0.2, 1.2, self.B).reshape(self.B, 1)
            return torch.hstack([partition[index] for index in [torch.randperm(self.B) for _ in range(self.p + 1)]])
        elif sample_type == 'fixed_X':
            partition = torch.linspace(-0.2, 1.2, self.M).reshape(self.M, 1).to(self.device)
            X, Y, Z = self.data_sa
            return torch.hstack([torch.kron(ones_N, partition), torch.kron(X, ones_M)])

    def train(self):

        X, Y, Z,kappa, gamma, llambda, delta = self.data_sa
        N = X.shape[0]
        m = self.M
        n = self.n
        ones_4 = torch.ones(4, 1).to(self.device)
        ones_m = torch.ones(m, 1).to(self.device)
        ones_N = torch.ones(self.N, 1).to(self.device)
        ones_B = torch.ones(self.BATCHSIZE, 1).to(self.device)
        ones_n = torch.ones(self.BATCHSIZE, 1).to(self.device)
        ones_mn = torch.ones(m * n, 1).to(self.device)

        a,beta_net = self.model()
        a,beta_net = a.to(self.device),beta_net.to(self.device)

        a.apply(self.weight_init())
        beta_net.apply(self.weight_init_beta())

        Parameters_a = [{"params": a.parameters()}]
        optimizer_a = torch.optim.Adam(Parameters_a, lr=1e-4)

        Parameters_beta = beta_net.parameters()
        optimizer_beta = torch.optim.RMSprop(Parameters_beta)
        lr_beta_init = optimizer_beta.param_groups[0]['lr']

        data_train0 = self.training_sample(self.sample_type)
        data_train0[:, [0]] = torch.rand(self.B, 1) - 0.5
        train_data = Data.TensorDataset(data_train0, torch.zeros(self.B, 1))
        train_loader = Data.DataLoader(dataset=train_data, batch_size=self.BATCHSIZE, shuffle=True)

        if self.v_sample == 'mesh':
            m = 100 + 1
            v = torch.linspace(0,1,m).reshape(m,1).to(self.device)
            h = 1 / (m - 1)
            W = (h*torch.diag(torch.hstack([0.5*torch.ones(1), torch.ones(m - 2), 0.5*torch.ones(1)]))).to(self.device)
        elif self.v_sample == 'mc':
            m = 100 + 1
            v = torch.rand(m,1) - 0.5
            v = v.to(self.device)

        writer = SummaryWriter()
        for epoch in range(self.EPOCH):
            for step, (data_train, _) in enumerate(train_loader):

                self.adjust_learning_rate(optimizer_beta, epoch, lr_beta_init)
                lr = optimizer_beta.param_groups[0]['lr']

                def fyz_vx(yzvx):
                    y = yzvx[:, [0]]
                    z = yzvx[:, [1]]
                    v = yzvx[:, [2]]
                    x = yzvx[:, 3:(3 + self.p)].clone()

                    pz = self.expit(x @ kappa + gamma * v)
                    py = self.expit(x @ llambda + beta_net(z) + delta * v)
                    return (z * pz + (1 - z) * (1 - pz)) * (y * py + (1 - y) * (1 - py))

                def S(yzvx):
                    y = yzvx[:, [0]]
                    z = yzvx[:, [1]]
                    v = yzvx[:, [2]]
                    x = yzvx[:, 3:(3 + self.p)].clone()

                    py = self.expit(x @ llambda + beta_net(z) + delta * v)
                    return (y - py) * z

                data_train = data_train.to(self.device)
                x = data_train[:, 1:(self.p + 1)].clone()  # n by p

                yz = torch.Tensor([[y, z] for y in range(2) for z in range(2)]).to(self.device)  # 4 by 2
                vx = torch.hstack([torch.kron(ones_n, v), torch.kron(x, ones_m)])
                yzvx = torch.hstack([torch.kron(yz, ones_mn), torch.kron(ones_4, vx)])

                p_yzvx = fyz_vx(yzvx).data.to(self.device)  # mn4 by 1
                s_yzvx = S(yzvx).data.to(self.device)  # mn4 by 1
                a_vx = torch.kron(ones_4, a(vx))  # 4mn by 1

                if self.v_sample == 'mesh':
                    num = torch.sum((p_yzvx * (a_vx - s_yzvx)).reshape(4, n, m) @ W, dim=2)  # 4 by n
                    denom = torch.sum(p_yzvx.reshape(4, n, m) @ W, dim=2)  # 4 by n
                elif self.v_sample == 'mc':
                    num = torch.mean((p_yzvx * (a_vx - s_yzvx)).reshape(4, n, m), dim=2)  # 4 by n
                    denom = torch.mean(p_yzvx.reshape(4, n, m), dim=2)  # 4 by n

                S_eff = num / (denom + 1e-20 * (denom == 0))  # 4 by n

                # sum over yz
                yzux = torch.hstack([torch.kron(yz, ones_n), torch.kron(ones_4, data_train)])  # 4n by 1
                p_yzux = fyz_vx(yzux).reshape(4, n)  # 4 by n

                sum_yz = torch.sum(S_eff * p_yzux, dim=0).reshape(n, 1)  # n by 1
                a_ux = a(data_train)  # n by 1
                loss_a = torch.mean((sum_yz - self.alpha * a_ux) ** 2)

                optimizer_a.zero_grad()
                loss_a.backward()
                optimizer_a.step()

                if epoch % self.alter_freq == 0:

                    vX = torch.hstack([torch.kron(ones_N, v), torch.kron(X, ones_m)])  # mN by 3
                    a_vX = a(vX).data.to(self.device)  # mN by 1
                    YZvX = torch.hstack([torch.kron(Y, ones_m), torch.kron(Z, ones_m), vX])  # mN by 5

                    s_YZvX = S(YZvX)  # mN by 1
                    p_YZvX = fyz_vx(YZvX)  # mN by 1

                    if self.v_sample == 'mesh':
                        num = torch.sum(((s_YZvX - a_vX) * p_YZvX).reshape(N, m) @ W, dim=1).reshape(N, 1)  # N
                        denom = torch.sum(p_YZvX.reshape(N, m) @ W, dim=1).reshape(N, 1)  # N
                    elif self.v_sample == 'mc':
                        num = torch.mean(((s_YZvX - a_vX) * p_YZvX).reshape(N, m), dim=1).reshape(N, 1)  # N
                        denom = torch.mean(p_YZvX.reshape(N, m), dim=1).reshape(N, 1)  # N

                    S_eff_vec = num / (denom + 1e-20 * (denom == 0))

                    loss_beta = torch.mean(S_eff_vec) ** 2

                    optimizer_beta.zero_grad()
                    loss_beta.backward()
                    optimizer_beta.step()

                if epoch % 50 == 0:
                    writer.add_scalar('loss_a', loss_a, global_step=epoch)
                    writer.add_scalar('loss_beta', loss_beta, global_step=epoch)
                    writer.add_scalar('beta', beta_net.weight, global_step=epoch)

        return beta_net.weight.item()