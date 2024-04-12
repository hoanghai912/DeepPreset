import numpy as np
from tensorboardX import SummaryWriter

class LossWriter(object):
    def __init__(self, _dir, board_names=['total', 'learning_preset', 'simulating_lr', 'g_loss', 'd_loss']):
        self.writer = SummaryWriter(_dir)
        self.names = board_names
        self.losses = np.zeros(len(self.names))
        self.nof_samples = 0

    def reset(self):
        self.losses = np.zeros(len(self.names))
        self.nof_samples = 0

    def update(self, loss_list):
        self.losses += np.array(loss_list)
        self.nof_samples += 1

    def get_mean_losses(self):
        return self.losses/self.nof_samples
    
    def finish(self, epoch):
        self.losses /= self.nof_samples
        for i, name in enumerate(self.names):
            self.add_scalar(name, self.losses[i], epoch + 1)
        self.reset()
    
    def add_scalar(self, name, yvalue, xvalues):
        self.writer.add_scalar(name, yvalue, xvalues)
    
    def close(self):
        self.writer.close()