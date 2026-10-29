import copy

import torch


class EMA:
    """Exponential moving average of model parameters, kept on the same device."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_params = dict(self.module.named_parameters())
        for name, p in model.named_parameters():
            if p.requires_grad:
                ema_params[name].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd)
