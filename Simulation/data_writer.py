import numpy as np
import os


class DataWriter(object):
    def __init__(self, data_write_len, data_path, init_t=0):
        self.online_data_buffer = []
        self.init_t = init_t
        self.data_cnt = 0
        self.data_write_len = data_write_len
        self.data_path = data_path
        if not os.path.exists("Simulation/OnlineData/"):
            os.mkdir("Simulation/OnlineData/")

    def set_init_t(self, init_t):
        self.init_t = init_t

    def subscribe_data(self, data, curr_t):
        self.online_data_buffer.append(data)  # appending data to buffer

        # saving data
        if curr_t > (self.data_cnt + 1) * self.data_write_len + self.init_t:
            save_path = self.data_path + 'online_data' + str(self.data_cnt) + '.npy'
            with open(save_path, 'wb') as f:
                np.save(f, np.array(self.online_data_buffer))
            self.online_data_buffer = []  # clear data buffer after writing to file
            self.data_cnt += 1
            return 1

        else:
            return 0

