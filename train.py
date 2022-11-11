from torch.nn import functional as F
from KNODE.models import *
import time
import torch
import os.path

train_verbose_header = '\033[33m' + "[Trainer] " + '\033[0m'  # yellow color


def sample_and_grow(ode_train, traj_list, epochs, lr, lookahead, device, plot_freq=50, step_skip=1, l2_lambda=5e-9, save_path=None, verbose=True):
    wd = 0  # (used to train initial model), when set to non-zero, the weights update even with zero gradients
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, ode_train.parameters()),
                                 lr=lr, weight_decay=wd)  # no regularization
    for i in range(epochs):
        for idx, true_traj in enumerate(traj_list):
            n_segments, _, n_state = true_traj.size()
            true_segments_list = []

            for j in range(0, n_segments - lookahead + 1, 1):
                j = j + i % 1
                true_sampled_segment = true_traj[j:j + lookahead]
                true_segments_list.append(true_sampled_segment)
            # concatenating all batches together
            obs = torch.cat(true_segments_list, 1).to(device)

            pred_traj = []
            ode_input = obs[0, :, :].unsqueeze(1).to(device)  # initial condition has size [1499, 1, 17]
            pred_traj.append(ode_input)

            for k in range(len(true_sampled_segment) - 1):
                z1 = ode_train(ode_input, Tensor(np.arange(2)).to(device)).squeeze(1)
                ode_input = torch.cat([z1[:, :13].unsqueeze(1), obs[k + 1, :, 13:].unsqueeze(1)], 2)
                pred_traj.append(ode_input)

            pred_traj = torch.transpose(torch.cat(pred_traj, 1), 0, 1)

            l2_norm = sum(p.pow(2.0).sum()
                          for p in ode_train.parameters())
            if idx == 0:
                loss = F.mse_loss(pred_traj[:, :, :13], obs[:, :, :13]) + l2_lambda * l2_norm
            else:
                loss += F.mse_loss(pred_traj[:, :, :13], obs[:, :, :13]) + l2_lambda * l2_norm

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        n_old_layers = ode_train.func.n_old_layers

        for m in range(n_old_layers):
            if str(ode_train.func.nn_model[m])[:6] == 'Linear':
                ode_train.func.nn_model[m].weight.grad *= 0
                ode_train.func.nn_model[m].bias.grad *= 0

        optimizer.step()

        if i % plot_freq == 0:
            if i == 400:  # reduce learning rate after some epochs
                lr = 0.01
            elif i == 1200:
                lr = 0.001
            optimizer.param_groups[0]['lr'] = lr

    if save_path is not None:  # saving model at the end of training
        torch.save({'ode_train': ode_train}, save_path)


def find_latest_data(data_path):
    """
    find the latest saved data and return its index
    """
    data_cnt = 0
    while os.path.exists(data_path + "online_data" + str(data_cnt) + ".npy"):
        data_cnt += 1
    return data_cnt


def train(save_path, model_to_train, training_data_path_list, device, epochs=80, training_noise=1e-8, nn_reg_lambda=5e-8, verbose=True):
    training_data = []
    for i, data_path in enumerate(training_data_path_list):
        with open(data_path, 'rb') as f:
            train_set = np.load(f)
        train_len = train_set.shape[0]
        train_set = Tensor(train_set.reshape([train_len, -1]))
        train_traj = train_set.detach().unsqueeze(1)
        train_traj = train_traj + torch.randn_like(train_traj) * np.sqrt(training_noise)
        training_data.append(train_traj[:, :, -17:])

    # Parameters
    torch.manual_seed(0)
    step_skip = 1
    LOOKAHEAD = 2
    LR = 0.01
    plot_freq = 50
    sample_and_grow(model_to_train,
                    training_data,
                    epochs,
                    LR,
                    LOOKAHEAD,
                    device=device,
                    plot_freq=plot_freq,
                    step_skip=step_skip,
                    l2_lambda=nn_reg_lambda,
                    save_path=save_path,
                    verbose=verbose)


def online_train(model_path, data_path, verbose=True):
    model_cnt = 1
    data_cnt = 0
    device = "cpu"
    torch.manual_seed(0)
    # initializing the first model
    ode_train = torch.load("KNODE/SavedModels/add_model_exp_weighting.pth", map_location=device)["ode_train"].to(device)
    ode_train.func.nn_pool_size = 3  # the number of neural networks to keep

    while not os.path.exists(data_path + "online_end.npy"):  # the simulator creates a file end.npy to signal end of sim

        training_data_path_list = [data_path + "online_data" + str(data_cnt) + ".npy"]
        # waiting for data from simulator
        while not os.path.exists(training_data_path_list[0]):
            # if simulation signals termination
            if os.path.exists(data_path + "online_end.npy"):
                return
            time.sleep(3)
            continue

        ode_train.func.cascade()
        save_path = model_path + "online_model" + str(model_cnt) + ".pth"

        train(save_path, ode_train, training_data_path_list, device)

        ode_train = torch.load(save_path)["ode_train"].to(device)
        model_cnt += 1
        data_cnt = find_latest_data(data_path)

    if verbose: print(train_verbose_header + "Max Cascade Reached. EXIT")
