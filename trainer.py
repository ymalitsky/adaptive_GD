import numpy as np
import time

import numpy.linalg as la
import matplotlib.pyplot as plt

from loss_functions import gradient, loss

class Trainer:
    """
    Base class for experiments with logistic regression. Provides methods
    for running optimization methods, saving the logs and plotting the results.
    """
    def __init__(self, t_max=np.inf, it_max=np.inf, output_size=500, tolerance=0):
        if t_max is np.inf and it_max is np.inf:
            it_max = 100
            print('The number of iterations is set to 100.')
        self.t_max = t_max
        self.it_max = it_max
        self.output_size = output_size
        self.first_run = True
        self.tolerance = tolerance
    
    def run(self, w0, X, y, l2):
        if self.first_run:
            self.init_run(w0, X, y, l2)
        else:
            self.ts = list(self.ts)
            self.its = list(self.its)
            self.ws = list(self.ws)
        self.first_run = False
        while (self.it < self.it_max) and (time.time() - self.t_start < self.t_max):
            sufficiently_big_gradient = self.compute_grad()
            if not sufficiently_big_gradient:
                break
            self.estimate_stepsize()
            self.w = self.step()

            self.save_checkpoint()

        self.ts = np.array(self.ts)
        self.its = np.array(self.its)
        self.ws = np.array(self.ws)
        
    def compute_grad(self):
        self.grad = gradient(self.w, self.X, self.y, self.l2)
        return la.norm(self.grad) > self.tolerance
        
    def estimate_stepsize(self):
        pass
        
    def step(self):
        pass
            
    def init_run(self, w0, X, y, l2):
        self.X = X
        self.y = y
        self.l2 = l2
        self.n, self.d = X.shape
        assert(len(w0) == self.d)
        assert(len(y) == self.n)
        self.w = w0.copy()
        self.ws = [w0.copy()]
        self.its = [0]
        self.ts = [0]
        self.it = 0
        self.t = 0
        self.t_start = time.time()
        self.time_progress = 0
        self.iterations_progress = 0
        self.max_progress = 0
        
    def save_checkpoint(self, first_iterations=10):
        self.it += 1
        self.t = time.time() - self.t_start
        self.time_progress = int((self.output_size - first_iterations) * self.t / self.t_max)
        self.iterations_progress = int((self.output_size - first_iterations) * (self.it / self.it_max))
        if (max(self.time_progress, self.iterations_progress) > self.max_progress) or (self.it <= first_iterations):
            self.update_logs()
        self.max_progress = max(self.time_progress, self.iterations_progress)
        
    def update_logs(self):        
        self.ws.append(self.w.copy())
        self.ts.append(self.t)
        self.its.append(self.it)
        
    def iterate_loss(self, w):
        return loss(w, self.X, self.y, self.l2)
    
    def compute_loss_on_iterates(self):
        self.losses = np.array([self.iterate_loss(w) for w in self.ws])
    
    def plot_losses(self, label='', marker=',', f_star=None, markevery=None):
        if self.losses is None:
            self.compute_loss_on_iterates()
        if f_star is None:
            f_star = np.min(self.losses)
        if markevery is None:
            markevery = len(self.losses) // 20
        plt.plot(self.its, self.losses - f_star, label=label, marker=marker, markevery=markevery)