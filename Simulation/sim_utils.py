import numpy as np
import os
import glob


def clean_dir():
    model_list = glob.glob("KNODE/SavedModels/online*")
    data_list = glob.glob("Simulation/OnlineData/online*")
    for model_path in model_list:
        os.remove(model_path)
    for data_path in data_list:
        os.remove(data_path)


def get_metrics(state, flat):
    x               = state['x']
    x_des           = flat['x']
    mse             = (np.square(x_des - x)).mean(axis=None)
    print('\nMSE:', mse)


def merge_dicts(dicts_in):
    dict_out = {}
    for k in dicts_in[0].keys():
        dict_out[k] = []
        for d in dicts_in:
            dict_out[k].append(d[k])
        dict_out[k] = np.array(dict_out[k])
    return dict_out


def sanitize_control_dic(control_dic):
    control_dic['cmd_thrust'] = np.asarray(control_dic['cmd_thrust'], np.float).ravel()
    control_dic['cmd_moment'] = np.asarray(control_dic['cmd_moment'], np.float).ravel()
    return control_dic


def sanitize_trajectory_dic(trajectory_dic):
    trajectory_dic['x'] = np.asarray(trajectory_dic['x'], np.float).ravel()
    trajectory_dic['x_dot'] = np.asarray(trajectory_dic['x_dot'], np.float).ravel()

    return trajectory_dic


def quat_dot(quat, omega):
    (q0, q1, q2, q3) = (quat[0], quat[1], quat[2], quat[3])
    G = np.array([[q3,  q2, -q1, -q0],
                  [-q2,  q3,  q0, -q1],
                  [q1, -q0,  q3, -q2]])
    quat_dot = 0.5 * G.T @ omega
    quat_err = np.sum(quat**2) - 1
    quat_err_grad = 2 * quat
    quat_dot = quat_dot - quat_err * quat_err_grad
    return quat_dot


def rotate_k(q):
    return np.array([2 * (q[0] * q[2] + q[1] * q[3]),
                     2 * (q[1] * q[2] - q[0] * q[3]),
                     1 - 2 * (q[0] ** 2 + q[1] ** 2)])


def hat_map(s):
    return np.array([[0, -s[2], s[1]],
                     [s[2], 0, -s[0]],
                     [-s[1], s[0], 0]])


def pack_state(state):
    s = np.zeros((13,))
    s[0:3] = state['x']
    s[3:6] = state['v']
    s[6:10] = state['q']
    s[10:13] = state['w']
    return s


def unpack_state(s):
    state = {'x': s[0:3], 'v': s[3:6], 'q': s[6:10], 'w': s[10:13]}
    return state