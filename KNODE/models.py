from KNODE.NODE import *


class QuadFullHybrid(ODEF):
    def __init__(self, quad_params, nn_pool_size, device):
        super(QuadFullHybrid, self).__init__()
        self.nn_model = nn.ModuleList([nn.Linear(17, 64),
                                       nn.Tanh(),
                                       nn.Linear(64, 13)])
        self.nn_pool_size = nn_pool_size
        torch.nn.init.xavier_uniform_(self.nn_model[0].weight)
        torch.nn.init.xavier_uniform_(self.nn_model[2].weight)

        self.device = device

        self.mass = torch.tensor([quad_params['mass']]).to(self.device)
        self.Ixx = torch.tensor([quad_params['Ixx']]).to(self.device)
        self.Iyy = torch.tensor([quad_params['Iyy']]).to(self.device)
        self.Izz = torch.tensor([quad_params['Izz']]).to(self.device)

        self.inertia = torch.diag(Tensor([self.Ixx, self.Iyy, self.Izz])).to(self.device)

    def forward(self, z):
        bs = z.size()[0]
        z = z.squeeze(1)
        x = z[:, :13]
        u = z[:, 13:]

        xdot = torch.cat([x[:, 3].unsqueeze(1), x[:, 4].unsqueeze(1), x[:, 5].unsqueeze(1)], 1)
        xdotdot = Tensor([0, 0, -9.81]).expand(bs, 3)
        pqr_vec = torch.cat([x[:, 10].unsqueeze(1), x[:, 11].unsqueeze(1), x[:, 12].unsqueeze(1)], 1)

        G_T_input1 = torch.cat([x[:, 9].unsqueeze(1), x[:, 8].unsqueeze(1),
                                -x[:, 7].unsqueeze(1), -x[:, 6].unsqueeze(1)], 1)
        G_T_input2 = torch.cat([-x[:, 8].unsqueeze(1), x[:, 9].unsqueeze(1),
                                x[:, 6].unsqueeze(1), -x[:, 7].unsqueeze(1)], 1)
        G_T_input3 = torch.cat([x[:, 7].unsqueeze(1), -x[:, 6].unsqueeze(1),
                                x[:, 9].unsqueeze(1), -x[:, 8].unsqueeze(1)], 1)
        G_transpose = torch.cat([G_T_input1.unsqueeze(2), G_T_input2.unsqueeze(2), G_T_input3.unsqueeze(2)], 2)

        quat_dot = 0.5 * torch.matmul(G_transpose, pqr_vec.unsqueeze(2)).squeeze(2)

        temp = -torch.cross(pqr_vec, torch.matmul(self.inertia.expand(bs, 3, 3), pqr_vec.unsqueeze(2)).squeeze(2), 1)

        pqr_dot = torch.matmul(torch.linalg.pinv(self.inertia).expand(bs, 3, 3), temp.unsqueeze(2)).squeeze(2)

        ode_without_u = torch.cat([xdot, xdotdot, quat_dot, pqr_dot], 1)

        xdotdot_u = torch.cat([2 * (x[:, 6] * x[:, 8] + x[:, 7] * x[:, 9]).unsqueeze(1),
                               2 * (x[:, 7] * x[:, 8] - x[:, 6] * x[:, 9]).unsqueeze(1),
                               (1 - 2 * (x[:, 6] ** 2 + x[:, 7] ** 2)).unsqueeze(1)], 1) / self.mass * u[:, 0].unsqueeze(1)

        pqrdot_u = torch.matmul(torch.linalg.pinv(self.inertia).expand(bs, 3, 3),
                                (torch.cat([u[:, 1].unsqueeze(1),
                                            u[:, 2].unsqueeze(1),
                                            u[:, 3].unsqueeze(1)], 1)).unsqueeze(2))

        u_component = torch.cat([torch.zeros_like(xdotdot_u),
                                 xdotdot_u,
                                 torch.zeros_like(quat_dot),
                                 pqrdot_u.squeeze(2)], 1)
        y_true_wou = ode_without_u + u_component
        y_true = torch.cat([y_true_wou, torch.zeros_like(quat_dot)], 1)

        temp = z

        for i, l in enumerate(self.nn_model):
            temp = l(temp)
            if (i + 1) % 3 == 0:
                # exponential weighting
                exp_weight = torch.exp(torch.tensor((i + 1 - len(self.nn_model))/3))
                y_true[:, :13] += exp_weight * temp
                if (i + 1) != len(self.nn_model):
                    temp = z

        return y_true.unsqueeze(1)


class QuadFullHybridAdd(QuadFullHybrid):
    def __init__(self, quad_params, nn_pool_size, device):
        super().__init__(quad_params, nn_pool_size, device)
        self.n_old_layers = 0

    def cascade(self):
        self.n_old_layers = len(self.nn_model)
        if self.n_old_layers == self.nn_pool_size * 3:
            self.n_old_layers = self.n_old_layers - 3
        new_layers = [nn.Linear(17, 64),
                      nn.Tanh(),
                      nn.Linear(64, 13)]

        for i in range(len(new_layers)):
            if str(new_layers[i])[:6] == 'Linear':
                torch.nn.init.xavier_uniform_(new_layers[i].weight)  # initialization
                torch.nn.init.zeros_(new_layers[i].bias)

        # adding new layers to the nn module
        self.nn_model.extend(new_layers)
        # only keep fixed number of layers
        self.nn_model = self.nn_model[-self.nn_pool_size*3:]