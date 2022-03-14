from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
import matplotlib.pyplot as plt
from env import WaveEnv


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self):
        return True
    
    def _on_rollout_end(self):
        # create test environment
        test_env = WaveEnv()

        # reset
        state = test_env.reset()

        # log initial norm of y and plot initial y
        self.logger.record('custom/init_norm_y', test_env.sim.norm_y())
        figure = plt.figure()
        figure.add_subplot().plot(test_env.sim.x, test_env.sim.y)
        self.logger.record("custom/init_y", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
        plt.close()
        
        # execute environment using trained policy until done
        done = False
        while not done:
            action = self.model.predict(state, deterministic=True)[0]
            state, reward, done, _ = test_env.step(action)

        # log final norm of y and plot final y
        self.logger.record('custom/done_norm_y', test_env.sim.norm_y())
        figure = plt.figure()
        figure.add_subplot().plot(test_env.sim.x, test_env.sim.y)
        self.logger.record("custom/done_y", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
        plt.close()
