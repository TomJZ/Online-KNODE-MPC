import numpy as np
import torch
from torch import Tensor
from torch import nn

class ODEF(nn.Module):
    def forward_with_grad(self, z, grad_outputs):
        """Compute f and a df/dz, a df/dp, a df/dt"""
        batch_size = z.shape[0]
        out = self.forward(z)

        a = grad_outputs
        adfdz, *adfdp = torch.autograd.grad(
            # concatenating tuples
            (out,), (z,) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )
        # grad method automatically sums gradients for batch items, we have to expand them back
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0) # unsqueeze(0) add dimension 1 to the position 0
            adfdp = adfdp.expand(batch_size, -1) / batch_size # passing -1 does not change dimension in that position
        return out, adfdz, adfdp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)

class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func, ode_solve, STEP_SIZE):
        assert isinstance(func, ODEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)

        with torch.no_grad():
            # initialize z to len of time and type of z0
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0
            # solving throughout time
            for i_t in range(time_len - 1):
                # z0 updated to next step
                z0 = ode_solve(z0, torch.abs(t[i_t+1]-t[i_t]), func, STEP_SIZE)
                z[i_t+1] = z0

        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        ctx.ode_solve = ode_solve
        ctx.STEP_SIZE = STEP_SIZE
        return z

    @staticmethod
    def backward(ctx, dLdz):
        """
        dLdz shape: time_len, batch_size, *z_shape
        """
        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)
        ode_solve = ctx.ode_solve
        STEP_SIZE = ctx.STEP_SIZE

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i):
            """
            tensors here are temporal slices
            t_i - is tensor with size: bs, 1
            aug_z_i - is tensor with size: bs, n_dim*2 + n_params + 1
            """
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2*n_dim]  # ignore parameters and time

            # Unflatten z and a
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)
            with torch.set_grad_enabled(True):
                z_i = z_i.detach().requires_grad_(True)
                func_eval, adfdz, adfdp = func.forward_with_grad(z_i, grad_outputs=a)  # bs, *z_shape
                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)

            # Flatten f and adfdz
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim)
            return torch.cat((func_eval, -adfdz, -adfdp), dim=1)

        dLdz = dLdz.view(time_len, bs, n_dim)  # flatten dLdz for convenience
        with torch.no_grad():
            ## Create placeholders for output gradients
            # Prev computed backwards adjoints to be adjusted by direct gradients
            adj_z = torch.zeros(bs, n_dim).to(dLdz)
            adj_p = torch.zeros(bs, n_params).to(dLdz)
            # In contrast to z and p we need to return gradients for all times
            adj_t = torch.zeros(time_len, bs, 1).to(dLdz)

            for i_t in range(time_len-1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                f_i = func(z_i).view(bs, n_dim)
                # Compute direct gradients
                dLdz_i = dLdz[i_t]

                # Adjusting adjoints with direct gradients
                adj_z += dLdz_i

                # Pack augmented variable
                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z)), dim=-1)

                # Solve augmented system backwards
                aug_ans = ode_solve(aug_z, torch.abs(t_i-t[i_t-1]), augmented_dynamics, -STEP_SIZE)

                # Unpack solved backwards augmented system
                adj_z[:] = aug_ans[:, n_dim:2*n_dim]
                adj_p[:] += aug_ans[:, 2*n_dim:2*n_dim + n_params]

                del aug_z, aug_ans

            ## Adjust 0 time adjoint with direct gradients
            # Compute direct gradients
            dLdz_0 = dLdz[0]

            # Adjust adjoints
            adj_z += dLdz_0
        #print("\nreturned grad:\n", adj_p)
        return adj_z.view(bs, *z_shape), None, adj_p, None, None, None

class NeuralODE(nn.Module):
    def __init__(self, func, ode_solve, STEP_SIZE):
        super(NeuralODE, self).__init__()
        assert isinstance(func, ODEF)
        self.func = func
        self.ode_solve = ode_solve
        self.STEP_SIZE = STEP_SIZE

    def forward(self, z0, t=Tensor([0., 1.]), return_whole_sequence=False):
        t = t.to(z0)
        z = ODEAdjoint.apply(z0, t, self.func.flatten_parameters(), self.func, self.ode_solve, self.STEP_SIZE)
        if return_whole_sequence:
            return z
        else:
            return z[-1]