import numpy as np

import numpy.linalg as la

from trainer import Trainer


class Gd(Trainer):
    """
    Gradient descent with constant learning rate.
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, lr, *args, **kwargs):
        super(Gd, self).__init__(*args, **kwargs)
        self.lr = lr
        
    def step(self):
        return self.w - self.lr * self.grad
    
    def init_run(self, *args, **kwargs):
        super(Gd, self).init_run(*args, **kwargs)
    
    
class Nesterov(Trainer):
    """
    Nesterov's accelerated gradient descent with constant learning rate.
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
        strongly_convex (boolean, optional): if true, uses the variant
            for strongly convex functions, which requires mu>0 (default: False)
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, lr, strongly_convex=False, mu=0, *args, **kwargs):
        super(Nesterov, self).__init__(*args, **kwargs)
        self.lr = lr
        if mu < 0:
            raise ValueError("Invalid mu: {}".format(mu))
        if strongly_convex and mu == 0:
            raise ValueError("""Mu must be larger than 0 for strongly_convex=True,
                             invalid value: {}""".format(mu))
        if strongly_convex:
            self.mu = mu
            kappa = (1/self.lr)/self.mu
            self.momentum = (np.sqrt(kappa)-1) / (np.sqrt(kappa)+1)
        self.strongly_convex = strongly_convex
        
    def step(self):
        if not self.strongly_convex:
            alpha_new = 0.5 * (1 + np.sqrt(1 + 4 * self.alpha ** 2))
            self.momentum = (self.alpha - 1) / alpha_new
            self.alpha = alpha_new
        self.w_nesterov_old = self.w_nesterov.copy()
        self.w_nesterov = self.w - self.lr * self.grad
        return self.w_nesterov + self.momentum * (self.w_nesterov - self.w_nesterov_old)
    
    def init_run(self, *args, **kwargs):
        super(Nesterov, self).init_run(*args, **kwargs)
        self.w_nesterov = self.w.copy()
        self.alpha = 1.
    
    
class Adgd(Trainer):
    """
    Adaptive gradient descent based on the local smoothness constant
    
    Arguments:
        eps (float): an estimate of 1 / L^2, where L is the global smoothness constant
    """
    def __init__(self, eps=0.0, lr0=None, *args, **kwargs):
        if not 0.0 <= eps:
            raise ValueError("Invalid eps: {}".format(eps))
        super(Adgd, self).__init__(*args, **kwargs)
        self.eps = eps
        self.lr0 = lr0
        
    def estimate_stepsize(self):
        L = la.norm(self.grad - self.grad_old) / la.norm(self.w - self.w_old)
        if np.isinf(self.theta):
            lr_new = 0.5 / L
        else:
            lr_new = min(np.sqrt(1 + self.theta) * self.lr, self.eps / self.lr + 0.5 / L)
        self.theta = lr_new / self.lr
        self.lr = lr_new
        
    def step(self):
        self.w_old = self.w.copy()
        self.grad_old = self.grad.copy()
        return self.w - self.lr * self.grad
        
    def init_run(self, *args, **kwargs):
        super(Adgd, self).init_run(*args, **kwargs)
        self.theta = np.inf
        grad = self.grad_func(self.w)
        if self.lr0 is None:
            self.lr0 = 1e-10
        self.lr = self.lr0
        self.lrs = [self.lr]
        self.w_old = self.w.copy()
        self.grad_old = grad
        self.w -= self.lr * grad
        self.save_checkpoint()
        
    def update_logs(self):
        super(Adgd, self).update_logs()
        self.lrs.append(self.lr)
        
        
class AdgdAccel(Trainer):
    """
    Adaptive gradient descent with heuristic Nesterov's acceleration
    Targeted at locally strongly convex functions, so by default uses
    estimation with min(sqrt(1 + theta_{k-1} / 2) * la_{k-1}, 0.5 / L_k)
    
    Arguments:
        a_lr (float, optional): increase parameter for learning rate (default: 0.5)
        a_mu (float, optional): increase parameter for strong convexity (default: 0.5)
        b_lr (float, optional): local smoothness scaling (default: 0.5)
        b_mu (float, optional): local strong convexity scaling (default: 0.5)
    """
    def __init__(self, a_lr=0.5, a_mu=0.5, b_lr=0.5, b_mu=0.5, *args, **kwargs):
        super(AdgdAccel, self).__init__(*args, **kwargs)
        self.a_lr = a_lr
        self.a_mu = a_mu
        self.b_lr = b_lr
        self.b_mu = b_mu
        
    def estimate_stepsize(self):
        L = la.norm(self.grad - self.grad_old) / la.norm(self.w - self.w_old)
        lr_new = min(np.sqrt(1 + self.a_lr * self.theta_lr) * self.lr, self.b_lr / L)
        self.theta_lr = lr_new / self.lr
        self.lr = lr_new
        mu_new = min(np.sqrt(1 + self.a_mu * self.theta_mu) * self.mu, self.b_lr * L)
        self.theta_mu = mu_new / self.mu
        self.mu = mu_new
        
    def step(self):
        self.w_old = self.w.copy()
        self.grad_old = self.grad.copy()
        momentum = (np.sqrt(1 / self.lr) - np.sqrt(self.mu)) / (np.sqrt(1 / self.lr) + np.sqrt(self.mu))
        self.w_nesterov_old = self.w_nesterov.copy()
        self.w_nesterov = self.w - self.lr * self.grad
        return self.w_nesterov + momentum * (self.w_nesterov - self.w_nesterov_old)
        
    def init_run(self, *args, **kwargs):
        super(AdgdAccel, self).init_run(*args, **kwargs)
        self.theta_lr = np.inf
        self.theta_mu = np.inf
        grad = self.grad_func(self.w)
        # The first estimate is normalized gradient with a small coefficient
        self.lr = 1e-5 / la.norm(grad)
        self.lrs = [self.lr]
        self.mu = 1 / self.lr
        self.w_old = self.w.copy()
        self.w_nesterov = self.w.copy()
        self.grad_old = grad
        self.w -= self.lr * grad
        self.save_checkpoint()
        
    def update_logs(self):
        super(AdgdAccel, self).update_logs()
        self.lrs.append(self.lr)

        
class Adagrad(Trainer):
    """
    Implement Adagrad from Duchi et. al, 2011
    "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"
    http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
    
    Arguments:
        primal_dual (boolean, optional): if true, uses the dual averaging method of Nesterov, 
            otherwise uses gradient descent update (default: False)
        eta (float, optional): learning rate scaling, but needs to be tuned to
            get better performance (default: 1)
        delta (float, optional): another learning rate parameter, slows down performance if
            chosen too large, otherwise requires tuning (default: 0)
    """
    def __init__(self, primal_dual=False, eta=1, delta=0, *args, **kwargs):
        super(Adagrad, self).__init__(*args, **kwargs)
        self.primal_dual = primal_dual
        self.eta = eta
        self.delta = delta
        
    def estimate_stepsize(self):
        self.s = np.sqrt(self.s ** 2 + self.grad ** 2)
        self.inv_lr = self.delta + self.s
        assert len(self.inv_lr) == len(self.w)
        
    def step(self):
        if self.primal_dual:
            self.sum_grad += self.grad
            return self.w0 - self.eta * np.divide(self.sum_grad, self.inv_lr, out=np.zeros_like(self.inv_lr), where=self.inv_lr != 0)
        else:
            return self.w - self.eta * np.divide(self.grad, self.inv_lr, out=np.zeros_like(self.inv_lr), where=self.inv_lr != 0)
        
    def init_run(self, *args, **kwargs):
        super(Adagrad, self).init_run(*args, **kwargs)
        self.w0 = self.w.copy()
        self.s = np.zeros(len(self.w))
        self.sum_grad = np.zeros(self.d)
        
        
class MirrorDescent(Trainer):
    """
    Gradient descent with constant learning rate.
    
    Arguments:
        lr (float, optional): an estimate of the inverse smoothness constant
    """
    def __init__(self, lr, mirror_step, *args, **kwargs):
        super(MirrorDescent, self).__init__(*args, **kwargs)
        self.lr = lr
        self.mirror_step = mirror_step
        
    def step(self):
        return self.mirror_step(self.w, self.lr, self.grad)
    
    def init_run(self, *args, **kwargs):
        super(MirrorDescent, self).init_run(*args, **kwargs)