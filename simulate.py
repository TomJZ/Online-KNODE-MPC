import copy
import numpy as np
import time
import os

from Simulation.system_change import system_change
from Simulation.sim_utils import sanitize_control_dic, sanitize_trajectory_dic, merge_dicts, get_metrics
from Simulation.quadrotor import instantiate_quadrotor
from Simulation.trajectory import instantiate_trajectory
from Simulation.control import instantiate_controller
from Simulation.data_writer import DataWriter


def simulate(t_final, desired_speed, length, model_path, data_path):
    sim_verbose_header = '\033[94m' + "[Simulator] " + '\033[0m'  # blue color

    quadrotor, initial_state = instantiate_quadrotor(length)
    trajectory = instantiate_trajectory(length, t_final, desired_speed)
    controller = instantiate_controller("KNODE/SavedModels/add_model_exp_weighting.pth")
    data_write_len = 0.15  # Length of data to write (in seconds)
    data_writer = DataWriter(data_write_len, data_path)
    change_dict = {'mass': [2.0, -0.015],
                   'mass2': [5.0, 0.025]}

    t_step = 1 / 500
    total_steps = int(t_final / t_step)
    time_stamps = [0]
    initial_state = {k: np.array(v) for k, v in initial_state.items()}
    state = [copy.deepcopy(initial_state)]
    flat = []
    control = []

    model_cnt = 1  # model count for online learning
    mass_change_t = change_dict['mass'][0]
    data_writer.set_init_t(mass_change_t)

    for cnt in range(total_steps):
        time_stamps.append(time_stamps[-1] + t_step)
        if cnt % (0.5 / t_step) == 0:
            print(sim_verbose_header + "Simulating {:.2f} sec".format(time_stamps[-1]))

        flat.append(sanitize_trajectory_dic(trajectory.update(time_stamps[-1])))

        if cnt % 25 == 0:
            ctrl_update = controller.update(state[-1], flat[-1])
            online_model_path = model_path + "online_model" + str(model_cnt) + ".pth"
            if cnt != 0:
                ret = controller.update_model(online_model_path)
                if ret == 1:
                    model_cnt += 1

        control.append(sanitize_control_dic(ctrl_update))

        state.append(quadrotor.update(state[-1], control[-1], t_step))
        system_change(quadrotor, cnt * t_step, change_dict)

        data_for_online = np.concatenate([np.hstack(list(state[-1].values())), ctrl_update['cmd_thrust'], ctrl_update['cmd_moment']], 0)
        # data writing
        if cnt * t_step > mass_change_t:
            if data_writer is not None:
                write_ret = data_writer.subscribe_data(data_for_online, cnt * t_step)

            if write_ret == 1:
                while not os.path.exists(online_model_path):  # if the node model is not ready, skip updating
                    time.sleep(3)

    # save the file with name online_end.npy to signal the end of simulation
    with open(data_path + 'online_end.npy', 'wb') as f:
        np.save(f, np.array([0]))

    # Results and metrics
    get_metrics(merge_dicts(state[:-1]), merge_dicts(flat))
