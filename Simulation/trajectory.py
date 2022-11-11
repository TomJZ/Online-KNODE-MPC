import numpy as np


class Trajectory(object):
    def __init__(self, points, desired_speed):
        self.points         = points
        self.desired_spd    = desired_speed

        self.time   = np.zeros((self.points.shape[0],))
        for i, d in enumerate(np.linalg.norm(np.diff(self.points, axis=0), axis=1)):
            self.time[i + 1] = self.time[i] + d/1.5

    def update(self, t):
        num_segments = len(self.points) - 1
        if num_segments > 0:
            segment_dists = self.points[1:(num_segments + 1), :] - self.points[0:num_segments, :]
            norm_dists = np.linalg.norm(segment_dists, axis=1)
            unit_vec = segment_dists / norm_dists[:, None]
            start_times = np.cumsum(norm_dists / self.desired_spd)

            if t < start_times[len(start_times) - 1]:
                idx = np.where(t <= start_times)[0]
                segment_num = idx[0]

                diff_time = t - start_times[segment_num]
                x_dot = self.desired_spd * unit_vec[segment_num, :]
                x = self.points[segment_num + 1, :] + x_dot * diff_time
            else:
                segment_num = num_segments - 1
                x_dot = np.zeros((3,))
                x = self.points[segment_num + 1, :]
        else:
            x_dot = np.zeros((3,))
            x = self.points

        output = {'x': x, 'x_dot': x_dot}
        return output


def instantiate_trajectory(length, t_final, desired_traj_speed):
    t_plot          = np.linspace(0, t_final+10.0, num=550)
    points          = np.stack((length*np.cos(t_plot), length*np.sin(t_plot), np.zeros((len(t_plot),))), axis=1)
    trajectory      = Trajectory(points, desired_traj_speed)
    return trajectory