import torch
from torch.optim.optimizer import Optimizer, required
from torch.distributions.normal import Normal


class DPSGD(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False, maximize=False,
                 l2_norm_clip=required, noise_multiplier=required, batch_size=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.l2_norm_clip = l2_norm_clip
        self.noise = Normal(0.0, noise_multiplier * l2_norm_clip / (batch_size ** 0.5))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        super(DPSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DPSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        total_norm = 0
        # Total gradient
        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None, group['params']):
                param_norm = p.grad.data.norm(2.)
                # item() -> Returns the value of this tensor as a standard Python number.
                total_norm += param_norm.item() ** 2.
            total_norm = total_norm ** (1. / 2.)

        # Gradient clipping
        clip_coef = self.l2_norm_clip / (total_norm + 1e-6)
        if clip_coef < 1:
            for group in self.param_groups:
                for p in filter(lambda p: p.grad is not None, group['params']):
                    p.grad.data.mul_(clip_coef)

        # Noise injection
        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None, group['params']):
                p.grad.data.add_(self.noise.sample(p.grad.data.size()).to(self.device))

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in filter(lambda p: p.grad is not None, group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                # Apply learning rate
                d_p.mul_(group['lr'])
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(d_p * -1)  # updated

        return loss
    
