import numpy as np
import pytorch_lightning as pl
import quadprog
import torch

from continual_learning.trainer import ContinualTrainer

class Strategy(pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__()

class GEM(Strategy):
    def __init__(self, memory_strength: float, ):
        super().__init__()

        self.memory_strength = memory_strength
        self.current_gradient = None

    def on_train_epoch_start(self, trainer: ContinualTrainer, pl_module: "pl.LightningModule") -> None:
        if trainer.task_id < 1:
            return

        gradient = [torch.cat([
            p.grad.detach().flatten()
            if p.grad is not None
            else torch.zeros(p.numel())
            for p in pl_module.parameters()
        ], dim=0)]

        self.current_gradient = torch.stack(gradient)  # (experiences, parameters)

    def on_after_backward(self, trainer: ContinualTrainer, pl_module: "pl.LightningModule") -> None:
        if trainer.task_id < 1:
            return

        gradient = torch.cat([
            p.grad.detach().flatten()
            if p.grad is not None
            else torch.zeros(p.numel())
            for p in pl_module.parameters()
        ], dim=0)

        to_project = (torch.mv(self.current_gradient, gradient) < 0).any()

        if to_project:
            v_star = self._solve_quadratic_programming(gradient).to(pl_module.device)

            num_pars = 0  # reshape v_star into the parameter matrices
            for p in pl_module.parameters():
                curr_pars = p.numel()
                if p.grad is not None:
                    p.grad.copy_(v_star[num_pars:num_pars + curr_pars].view(p.size()))
                num_pars += curr_pars

            if num_pars != v_star.numel():
                raise ValueError('Error in projecting gradient')

    def _solve_quadratic_programming(self, gradient) -> torch.Tensor:
        memories_np = self.current_gradient.cpu().double().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + self.memory_strength
        v = quadprog.solve_qp(P, q, G, h)[0]
        v_star = np.dot(v, memories_np) + gradient_np

        return torch.from_numpy(v_star).float()
