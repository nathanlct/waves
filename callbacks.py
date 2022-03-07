from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_training_start(self):
        self.env = self.training_env.envs[0]

    def _on_step(self):
        for infos in self.locals['infos']:
            for k, v in infos.items():
                if k not in ['episode', 'terminal_observation']:
                    self.logger.record(f'custom/{k}', v)
        return True
