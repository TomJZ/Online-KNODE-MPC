import numpy as np
from multiprocessing import Process
from Simulation.sim_utils import clean_dir
from simulate import simulate
from train import online_train

if __name__ == "__main__":
    """
    Initialization
    """
    np.random.seed(3)
    # Set simulation duration [s], desired speed [m/s], trajectory parameter [m]
    sim_duration, desired_speed, length = 8, 1.2, 3
    """
    Multi-process online training and simulation
    """
    clean_dir()
    model_path = "KNODE/SavedModels/"
    data_path = "Simulation/OnlineData/"

    trainer = Process(target=online_train, args=(model_path, data_path))
    simulator = Process(target=simulate, args=(sim_duration, desired_speed, length, model_path, data_path))

    trainer.start()
    simulator.start()

    trainer.join()
    simulator.join()
