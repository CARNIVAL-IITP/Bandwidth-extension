import json
import pickle

from torch import Tensor

import pytorch_lightning as pl
from copy import deepcopy

from typing import Optional, Union, Dict, Any

import pytorch_lightning as pl
import torch
from overrides import overrides
from pytorch_lightning.utilities import rank_zero_only
from continual_learning.trainer import ContinualTrainer
from models.continual_TUNet import ContinualTUNet
from torch.utils.data import DataLoader

def read_json_file(path: str) -> list:
    with open(path, mode='r') as f:
        data = json.load(f)
    return data


def save_json_file(data: list, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(data, f)


def pickle_dump(data: list, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def pickle_load(path: str) -> Tensor:
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data

class EWC_Callback(pl.Callback):
    def on_train_epoch_end(self, trainer:ContinualTrainer, pl_module: ContinualTUNet, train_dataloader: DataLoader):
        from continual_learning.strategy.ewc import EWC
        self.fisher_matrix = {}
        self.saved_params = {}
        for n, p in self.saved_params.items():
            t = torch.zeros_like(p.data)
            self.fisher_matrix[n] = t

        self.train_dataloader = train_dataloader
        pl_module.ewc_mode = True
        pl_module.fisher_matrix = self.fisher_matrix
        trainer.test(pl_module, train_dataloader)
        # pl_module.ewc_mode = False

        for n in self.fisher_matrix:
            self.fisher_matrix[n] /= len(self.train_dataset)

        self.ewc = EWC(ewc_lambda=0.1)
        self.ewc.calculate_importances(trainer, self, self.datamodule.train_dataloader)
        # do something with all training_step outputs, for example:


import contextlib
import copy
import os
import threading
from typing import Any, Dict, Iterable

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_info

class EMA(pl.Callback):
    """
    Implements Exponential Moving Averaging (EMA).
    When training a model, this callback will maintain moving averages of the trained parameters.
    When evaluating, we use the moving averages copy of the trained parameters.
    When saving, we save an additional set of parameters with the prefix `ema`.
    Args:
        decay: The exponential decay used when calculating the moving average. Has to be between 0-1.
        validate_original_weights: Validate the original weights, as apposed to the EMA weights.
        every_n_steps: Apply EMA every N steps.
        cpu_offload: Offload weights to CPU.
    """

    def __init__(
        self, decay: float, validate_original_weights: bool = False, every_n_steps: int = 1, cpu_offload: bool = False,
    ):
        if not (0 <= decay <= 1):
            raise MisconfigurationException("EMA decay value must be between 0 and 1")
        self.decay = decay
        self.validate_original_weights = validate_original_weights
        self.every_n_steps = every_n_steps
        self.cpu_offload = cpu_offload

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        device = pl_module.device if not self.cpu_offload else torch.device('cpu')
        trainer.optimizers = [
            EMAOptimizer(
                optim,
                device=device,
                decay=self.decay,
                every_n_steps=self.every_n_steps,
                current_step=trainer.global_step,
            )
            for optim in trainer.optimizers
            if not isinstance(optim, EMAOptimizer)
        ]

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_validate_ema_weights(trainer):
            self.swap_model_weights(trainer)

    def _should_validate_ema_weights(self, trainer: "pl.Trainer") -> bool:
        return not self.validate_original_weights and self._ema_initialized(trainer)

    def _ema_initialized(self, trainer: "pl.Trainer") -> bool:
        return any(isinstance(optimizer, EMAOptimizer) for optimizer in trainer.optimizers)

    def swap_model_weights(self, trainer: "pl.Trainer", saving_ema_model: bool = False):
        for optimizer in trainer.optimizers:
            assert isinstance(optimizer, EMAOptimizer)
            optimizer.switch_main_parameter_weights(saving_ema_model)

    @contextlib.contextmanager
    def save_ema_model(self, trainer: "pl.Trainer"):
        """
        Saves an EMA copy of the model + EMA optimizer states for resume.
        """
        self.swap_model_weights(trainer, saving_ema_model=True)
        try:
            yield
        finally:
            self.swap_model_weights(trainer, saving_ema_model=False)

    @contextlib.contextmanager
    def save_original_optimizer_state(self, trainer: "pl.Trainer"):
        for optimizer in trainer.optimizers:
            assert isinstance(optimizer, EMAOptimizer)
            optimizer.save_original_optimizer_state = True
        try:
            yield
        finally:
            for optimizer in trainer.optimizers:
                optimizer.save_original_optimizer_state = False

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        checkpoint_callback = trainer.checkpoint_callback

        # use the connector as NeMo calls the connector directly in the exp_manager when restoring.
        connector = trainer._checkpoint_connector
        ckpt_path = connector.resume_checkpoint_path

        if ckpt_path and checkpoint_callback is not None and 'NeMo' in type(checkpoint_callback).__name__:
            ext = checkpoint_callback.FILE_EXTENSION
            if ckpt_path.endswith(f'-EMA{ext}'):
                rank_zero_info(
                    "loading EMA based weights. "
                    "The callback will treat the loaded EMA weights as the main weights"
                    " and create a new EMA copy when training."
                )
                return
            ema_path = ckpt_path.replace(ext, f'-EMA{ext}')
            if os.path.exists(ema_path):
                ema_state_dict = torch.load(ema_path, map_location=torch.device('cpu'))

                checkpoint['optimizer_states'] = ema_state_dict['optimizer_states']
                del ema_state_dict
                rank_zero_info("EMA state has been restored.")
            else:
                raise MisconfigurationException(
                    "Unable to find the associated EMA weights when re-loading, "
                    f"training will start with new EMA weights. Expected them to be at: {ema_path}",
                )
            


@torch.no_grad()
def ema_update(ema_model_tuple, current_model_tuple, decay):
    torch._foreach_mul_(ema_model_tuple, decay)
    torch._foreach_add_(
        ema_model_tuple, current_model_tuple, alpha=(1.0 - decay),
    )


def run_ema_update_cpu(ema_model_tuple, current_model_tuple, decay, pre_sync_stream=None):
    if pre_sync_stream is not None:
        pre_sync_stream.synchronize()

    ema_update(ema_model_tuple, current_model_tuple, decay)


class EMAOptimizer(torch.optim.Optimizer):
    r"""
    EMAOptimizer is a wrapper for torch.optim.Optimizer that computes
    Exponential Moving Average of parameters registered in the optimizer.
    EMA parameters are automatically updated after every step of the optimizer
    with the following formula:
        ema_weight = decay * ema_weight + (1 - decay) * training_weight
    To access EMA parameters, use ``swap_ema_weights()`` context manager to
    perform a temporary in-place swap of regular parameters with EMA
    parameters.
    Notes:
        - EMAOptimizer is not compatible with APEX AMP O2.
    Args:
        optimizer (torch.optim.Optimizer): optimizer to wrap
        device (torch.device): device for EMA parameters
        decay (float): decay factor
    Returns:
        returns an instance of torch.optim.Optimizer that computes EMA of
        parameters
    Example:
        model = Model().to(device)
        opt = torch.optim.Adam(model.parameters())
        opt = EMAOptimizer(opt, device, 0.9999)
        for epoch in range(epochs):
            training_loop(model, opt)
            regular_eval_accuracy = evaluate(model)
            with opt.swap_ema_weights():
                ema_eval_accuracy = evaluate(model)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        decay: float = 0.9999,
        every_n_steps: int = 1,
        current_step: int = 0,
    ):
        self.optimizer = optimizer
        self.decay = decay
        self.device = device
        self.current_step = current_step
        self.every_n_steps = every_n_steps
        self.save_original_optimizer_state = False

        self.first_iteration = True
        self.rebuild_ema_params = True
        self.stream = None
        self.thread = None

        self.ema_params = ()
        self.in_saving_ema_model_context = False

    def all_parameters(self) -> Iterable[torch.Tensor]:
        return (param for group in self.param_groups for param in group['params'])

    def step(self, closure=None, **kwargs):
        self.join()

        if self.first_iteration:
            if any(p.is_cuda for p in self.all_parameters()):
                self.stream = torch.cuda.Stream()

            self.first_iteration = False

        if self.rebuild_ema_params:
            opt_params = list(self.all_parameters())

            self.ema_params += tuple(
                copy.deepcopy(param.data.detach()).to(self.device) for param in opt_params[len(self.ema_params) :]
            )
            self.rebuild_ema_params = False

        loss = self.optimizer.step(closure)

        if self._should_update_at_step():
            self.update()
        self.current_step += 1
        return loss

    def _should_update_at_step(self) -> bool:
        return self.current_step % self.every_n_steps == 0

    @torch.no_grad()
    def update(self):
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.stream):
            current_model_state = tuple(
                param.data.to(self.device, non_blocking=True) for param in self.all_parameters()
            )

            if self.device.type == 'cuda':
                ema_update(self.ema_params, current_model_state, self.decay)

        if self.device.type == 'cpu':
            self.thread = threading.Thread(
                target=run_ema_update_cpu, args=(self.ema_params, current_model_state, self.decay, self.stream,),
            )
            self.thread.start()

    def swap_tensors(self, tensor1, tensor2):
        tmp = torch.empty_like(tensor1)
        tmp.copy_(tensor1)
        tensor1.copy_(tensor2)
        tensor2.copy_(tmp)

    def switch_main_parameter_weights(self, saving_ema_model: bool = False):
        self.join()
        self.in_saving_ema_model_context = saving_ema_model
        for param, ema_param in zip(self.all_parameters(), self.ema_params):
            self.swap_tensors(param.data, ema_param)

    @contextlib.contextmanager
    def swap_ema_weights(self, enabled: bool = True):
        r"""
        A context manager to in-place swap regular parameters with EMA
        parameters.
        It swaps back to the original regular parameters on context manager
        exit.
        Args:
            enabled (bool): whether the swap should be performed
        """

        if enabled:
            self.switch_main_parameter_weights()
        try:
            yield
        finally:
            if enabled:
                self.switch_main_parameter_weights()

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def join(self):
        if self.stream is not None:
            self.stream.synchronize()

        if self.thread is not None:
            self.thread.join()

    def state_dict(self):
        self.join()

        if self.save_original_optimizer_state:
            return self.optimizer.state_dict()

        # if we are in the context of saving an EMA model, the EMA weights are in the modules' actual weights
        ema_params = self.ema_params if not self.in_saving_ema_model_context else list(self.all_parameters())
        state_dict = {
            'opt': self.optimizer.state_dict(),
            'ema': ema_params,
            'current_step': self.current_step,
            'decay': self.decay,
            'every_n_steps': self.every_n_steps,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.join()

        self.optimizer.load_state_dict(state_dict['opt'])
        self.ema_params = tuple(param.to(self.device) for param in copy.deepcopy(state_dict['ema']))
        self.current_step = state_dict['current_step']
        self.decay = state_dict['decay']
        self.every_n_steps = state_dict['every_n_steps']
        self.rebuild_ema_params = False

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)
        self.rebuild_ema_params = True


# # class EMA(pl.Callback):
#     """Implements EMA (exponential moving average) to any kind of model.
#     EMA weights will be used during validation and stored separately from original model weights.

#     How to use EMA:
#         - Sometimes, last EMA checkpoint isn't the best as EMA weights metrics can show long oscillations in time. See
#           https://github.com/rwightman/pytorch-image-models/issues/102
#         - Batch Norm layers and likely any other type of norm layers doesn't need to be updated at the end. See
#           discussions in: https://github.com/rwightman/pytorch-image-models/issues/106#issuecomment-609461088 and
#           https://github.com/rwightman/pytorch-image-models/issues/224
#         - For object detection, SWA usually works better. See   https://github.com/timgaripov/swa/issues/16

#     Implementation detail:
#         - See EMA in Pytorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning/issues/10914
#         - When multi gpu, we broadcast ema weights and the original weights in order to only hold 1 copy in memory.
#           This is specially relevant when storing EMA weights on CPU + pinned memory as pinned memory is a limited
#           resource. In addition, we want to avoid duplicated operations in ranks != 0 to reduce jitter and improve
#           performance.
#     """
#     def __init__(self, decay: float = 0.9999, ema_device: Optional[Union[torch.device, str]] = None, pin_memory=True):
#         super().__init__()
#         self.decay = decay
#         self.ema_device: str = f"{ema_device}" if ema_device else None  # perform ema on different device from the model
#         self.ema_pin_memory = pin_memory if torch.cuda.is_available() else False  # Only works if CUDA is available
#         self.ema_state_dict: Dict[str, torch.Tensor] = {}
#         self.original_state_dict = {}
#         self._ema_state_dict_ready = False

#     @staticmethod
#     def get_state_dict(pl_module: pl.LightningModule):
#         """Returns state dictionary from pl_module. Override if you want filter some parameters and/or buffers out.
#         For example, in pl_module has metrics, you don't want to return their parameters.
        
#         code:
#             # Only consider modules that can be seen by optimizers. Lightning modules can have others nn.Module attached
#             # like losses, metrics, etc.
#             patterns_to_ignore = ("metrics1", "metrics2")
#             return dict(filter(lambda i: i[0].startswith(patterns), pl_module.state_dict().items()))
#         """
#         return pl_module.state_dict()
        
#     @overrides
#     def on_train_start(self, trainer: "pl.Trainer", pl_module: pl.LightningModule) -> None:
#         # Only keep track of EMA weights in rank zero.
#         if not self._ema_state_dict_ready and pl_module.global_rank == 0:
#             self.ema_state_dict = deepcopy(self.get_state_dict(pl_module))
#             if self.ema_device:
#                 self.ema_state_dict = {k: tensor.to(device=self.ema_device) for k, tensor in self.ema_state_dict.items()}

#             if self.ema_device == "cpu" and self.ema_pin_memory:
#                 self.ema_state_dict = {k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()}

#         self._ema_state_dict_ready = True

#     @rank_zero_only
#     def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, *args, **kwargs) -> None:
#         # Update EMA weights
#         with torch.no_grad():
#             for key, value in self.get_state_dict(pl_module).items():
#                 ema_value = self.ema_state_dict[key]
#                 ema_value.copy_(self.decay * ema_value + (1. - self.decay) * value, non_blocking=True)

#     @overrides
#     def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
#         if not self._ema_state_dict_ready:
#             return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

#         self.original_state_dict = deepcopy(self.get_state_dict(pl_module))
#         pl_module.trainer.training_type_plugin.broadcast(self.ema_state_dict, 0)
#         assert self.ema_state_dict.keys() == self.original_state_dict.keys(), \
#             f"There are some keys missing in the ema static dictionary broadcasted. " \
#             f"They are: {self.original_state_dict.keys() - self.ema_state_dict.keys()}"
#         pl_module.load_state_dict(self.ema_state_dict, strict=False)

#         if pl_module.global_rank > 0:
#             # Remove ema state dict from the memory. In rank 0, it could be in ram pinned memory.
#             self.ema_state_dict = {}

#     @overrides
#     def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
#         if not self._ema_state_dict_ready:
#             return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

#         # Replace EMA weights with training weights
#         pl_module.load_state_dict(self.original_state_dict, strict=False)

#     @overrides
#     def on_save_checkpoint(
#         self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
#     ) -> dict:
#         return {"ema_state_dict": self.ema_state_dict, "_ema_state_dict_ready": self._ema_state_dict_ready}

#     @overrides
#     def on_load_checkpoint(
#         self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict[str, Any]
#     ) -> None:
#         self._ema_state_dict_ready = callback_state["_ema_state_dict_ready"]
#         self.ema_state_dict = callback_state["ema_state_dict"]