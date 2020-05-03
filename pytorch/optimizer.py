import torch
import numpy as np

from torch.optim.optimizer import Optimizer, required

    
class Adsgd(Optimizer):
    r"""
    Adaptive SGD with estimation of the local smoothness (curvature).
    Based on https://arxiv.org/abs/1910.09529
    """
    def __init__(self, params, lr=0.2, amplifier=0.02, theta=np.inf, damping=1, eps=1e-5, weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid initial learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, amplifier=amplifier, theta=theta, damping=damping,
                        option=option, eps=eps, weight_decay=weight_decay)
        super(Adsgd, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adsgd, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('lr', 0.2)
            group.setdefault('amplifier', 0.02)
            group.setdefault('damping', 1)
            group.setdefault('theta', np.inf)
                
    def compute_dif_norms(self, prev_optimizer=required):
        for group, prev_group in zip(self.param_groups, prev_optimizer.param_groups):
            grad_dif_norm = 0
            param_dif_norm = 0
            for p, prev_p in zip(group['params'], prev_group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                prev_d_p = prev_p.grad.data
                grad_dif_norm += (d_p - prev_d_p).norm().item() ** 2
                param_dif_norm += (p.data - prev_p.data).norm().item() ** 2
            group['grad_dif_norm'] = np.sqrt(grad_dif_norm)
            group['param_dif_norm'] = np.sqrt(param_dif_norm)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        
        # TODO: use closure to compute gradient difference
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            eps = group['eps']
            lr = group['lr']
            damping = group['damping']
            option = group['option']
            amplifier = group['amplifier']
            theta = group['theta']
            grad_dif_norm = group['grad_dif_norm']
            param_dif_norm = group['param_dif_norm']
            lr_new = min(lr * np.sqrt(1 + amplifier * theta), param_dif_norm / (damping * grad_dif_norm)) + eps
            theta = lr_new / lr
            group['theta'] = theta
            group['lr'] = lr_new
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group['weight_decay'] != 0:
                    d_p.add_(group['weight_decay'], p.data)
                p.data.add_(-lr_new, d_p)
        return loss