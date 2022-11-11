import numpy as np
import scipy.integrate
from Simulation.crazyflie_params import quad_params
from Simulation.sim_utils import quat_dot, rotate_k, hat_map, pack_state, unpack_state


class Quadrotor(object):
    def __init__(self):
        self.mass = quad_params['mass']
        self.inertia = np.diag(np.array([quad_params['Ixx'], quad_params['Iyy'], quad_params['Izz']]))
        self.inv_inertia = np.linalg.inv(self.inertia)

    def update(self, state, control, t_step):
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, control)

        sol = scipy.integrate.solve_ivp(s_dot_fn, (0, t_step), pack_state(state), first_step=t_step)
        state = unpack_state(sol['y'][:, -1])
        return state

    def _s_dot_fn(self, t, s, control):
        s_dot = np.zeros((13,))
        s_dot[0:3] = s[3:6]
        s_dot[3:6] = np.array([0, 0, -9.81]) + (control['cmd_thrust'] * rotate_k(s[6:10]))/self.mass
        s_dot[6:10] = quat_dot(s[6:10], s[10:13])
        s_dot[10:13] = self.inv_inertia @ (control['cmd_moment'] - hat_map(s[10:13]) @ (self.inertia @ s[10:13]))

        return s_dot


def instantiate_quadrotor(length):
    quadrotor       = Quadrotor()
    initial_state = {'x': np.array([length, 0, 0]),
                     'v': np.array([0, 0, 0]),
                     'q': np.array([0, 0., 0., 1.]),
                     'w': np.zeros(3, )}

    return quadrotor, initial_state
