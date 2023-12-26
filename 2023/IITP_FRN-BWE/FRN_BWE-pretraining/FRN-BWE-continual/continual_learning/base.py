import logging
import os, time
import traceback
from abc import abstractmethod
from typing import Optional, List, Union, Any, Iterable

import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
import wandb
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from torch.utils.data import DataLoader

from continual_learning.trainer import ContinualTrainer
from continual_learning.strategy.ewc import EWC
from continual_learning.strategy.gem import GEM
# from continual_ranking.dpr.data.data_module import DataModule
from utils.hydra_utls import EMA

from dataset.dataset_FRN_MSM_aggressive import TrainDataset_FRN_MSM_aggressive
from dataset.dataset_FRN_MSM_clean import TrainDataset_FRN_MSM_clean
from dataset.dataset_FRN_MSM_noisy import TrainDataset_FRN_MSM_noisy
from dataset.dataset_FRN_HB_BWE import TrainDataset_FRN_HB_BWE
from dataset.dataset_FRN_NAE import TrainDataset_FRN_NAE
from dataset.dataset_FRN_NB_BWE import TrainDataset_FRN_NB_BWE

from dataset.data_module import DataModule_FRN_HB_BWE, DataModule_FRN_NB_BWE, DataModule_FRN_NAE
from dataset.data_module import DataModule_FRN_MSM_aggressive, DataModule_FRN_MSM_clean, DataModule_FRN_MSM_noisy

from models.continual_FRN import ContinualFRN
from models.continual_FRN_film import ContinualFRN_film
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging as SWA
from utils.tblogger import TensorBoardLoggerExpanded
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
from pynvml import *
# from dataset.original_data_module import DataModule

from config_folder.config_FRN_HB_BWE import CONFIG

logger = logging.getLogger(__name__)


class Base:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.fast_dev_run = cfg.fast_dev_run
        self.logging_on = cfg.logging_on and not self.fast_dev_run
        self.experiment_name = cfg.experiment.name

        self.model: Optional[ContinualFRN] = None
        self.train_dataloader: Union[DataLoader, List[DataLoader]] = []
        self.val_dataloader: Union[DataLoader, List[DataLoader]] = []
        self.trainer: Optional[Union[ContinualTrainer, Any]] = None
        self.strategies: Optional[Iterable[pl.Callback]] = None
        self.tensorboard_logger = TensorBoardLoggerExpanded()

        self.loggers: List[Logger] = []
        self._early_stopping = None
        self.callbacks: List[pl.Callback] = []

        self.ewc: Optional[EWC] = None

    def setup_model(self) -> None:
        logger.info('Setting up model')
        if CONFIG.MODEL.model_name == 'FRN-baseline':
            self.model = ContinualFRN(
                train_dataset=self.train_dataset, 
                val_dataset= self.val_dataset)
        elif CONFIG.MODEL.model_name == 'FRN-FiLM':
            self.model = ContinualFRN_film(
                train_dataset=self.train_dataset, 
                val_dataset= self.val_dataset)
        
    def setup_loggers(self) -> None:
        tensorboard_logger = self.tensorboard_logger
        self.loggers.append(tensorboard_logger)
        csv_logger = CSVLogger(
            'csv',
            name = self.cfg.project_name,
            version = self.experiment_name
        )
        self.loggers.append(csv_logger)

    def setup_callbacks(self) -> None:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', verbose=True,
                                          filename='frn-{epoch:02d}-{val_loss:.4f}', save_weights_only=False,
                                          save_on_train_epoch_end=True)
        self.callbacks = [
        lr_monitor, checkpoint_callback
        ]

    def prepare_FRN_MSM_clean_dataloaders(self) -> None:
        logger.info('Setting up dataloaders')
        self.datamodule = DataModule_FRN_MSM_clean()
        self.train_dataloader = self.datamodule.train_dataloader()
        self.val_dataloader = self.datamodule.val_dataloader()
        self.train_dataset = TrainDataset_FRN_MSM_clean('train')
        self.val_dataset = TrainDataset_FRN_MSM_clean('val')
        self.train_length = len(self.train_dataset)
        self.val_length = len(self.val_dataset)

    def prepare_FRN_MSM_noisy_dataloaders(self) -> None:
        logger.info('Setting up dataloaders')
        self.datamodule = DataModule_FRN_MSM_noisy()
        self.train_dataloader = self.datamodule.train_dataloader()
        self.val_dataloader = self.datamodule.val_dataloader()
        self.train_dataset = TrainDataset_FRN_MSM_noisy('train')
        self.val_dataset = TrainDataset_FRN_MSM_noisy('val')
        self.train_length = len(self.train_dataset)
        self.val_length = len(self.val_dataset)

    def prepare_FRN_MSM_aggressive_dataloaders(self) -> None:
        logger.info('Setting up dataloaders')
        self.datamodule = DataModule_FRN_MSM_aggressive()
        self.train_dataloader = self.datamodule.train_dataloader()
        self.val_dataloader = self.datamodule.val_dataloader()
        self.train_dataset = TrainDataset_FRN_MSM_aggressive('train')
        self.val_dataset = TrainDataset_FRN_MSM_aggressive('val')
        self.train_length = len(self.train_dataset)
        self.val_length = len(self.val_dataset)

    def prepare_FRN_NB_BWE_dataloaders(self) -> None:
        logger.info('Setting up dataloaders')
        self.datamodule = DataModule_FRN_NB_BWE()
        self.train_dataloader = self.datamodule.train_dataloader()
        self.val_dataloader = self.datamodule.val_dataloader()
        self.train_dataset = TrainDataset_FRN_NB_BWE('train')
        self.val_dataset = TrainDataset_FRN_NB_BWE('val')
        self.train_length = len(self.train_dataset)
        self.val_length = len(self.val_dataset)

    def prepare_FRN_NAE_dataloaders(self) -> None:
        logger.info('Setting up dataloaders')
        self.datamodule = DataModule_FRN_NAE()
        self.train_dataloader = self.datamodule.train_dataloader()
        self.val_dataloader = self.datamodule.val_dataloader()
        self.train_dataset = TrainDataset_FRN_NAE('train')
        self.val_dataset = TrainDataset_FRN_NAE('val')
        self.train_length = len(self.train_dataset)
        self.val_length = len(self.val_dataset)

    def prepare_FRN_HB_BWE_dataloaders(self) -> None:
        logger.info('Setting up dataloaders')
        self.datamodule = DataModule_FRN_HB_BWE()
        self.train_dataloader = self.datamodule.train_dataloader()
        self.val_dataloader = self.datamodule.val_dataloader()
        self.train_dataset = TrainDataset_FRN_HB_BWE('train')
        self.val_dataset = TrainDataset_FRN_HB_BWE('val')
        self.train_length = len(self.train_dataset)
        self.val_length = len(self.val_dataset)

    def setup_trainer(self) -> None:
        logger = TensorBoardLoggerExpanded()
        # logger.__init__()
        # logger.info('Setting up trainer')
        self.trainer = ContinualTrainer(
            gradient_clip_val = self.cfg.clipping_val,
            max_epochs = 100, #self.cfg.max_epochs,
            accelerator = "gpu",
            # devices = [0, 1],
            # devices = [0],
            devices = [1],
            callbacks=self.callbacks,
        )

    def setup_base_strategies(self) -> None:
        if CONFIG.TRAIN.pretraining.strategy == 'regularizer':
            self.model.regularizer_mode = True
        elif CONFIG.TRAIN.pretraining.strategy == 'ewc':
            self.ewc = EWC(CONFIG.TRAIN.pretraining.ewc_lambda)
            strategy = self.ewc
            self.model.ewc = self.ewc
            self.callbacks.append(strategy)
            self.model.ewc_mode = True
        elif CONFIG.TRAIN.pretraining.strategy == 'ema':
            strategy = EMA(decay = CONFIG.TRAIN.pretraining.ema_decay)
            self.callbacks.append(strategy)
            self.model.ema_mode = True
        elif CONFIG.TRAIN.pretraining.strategy == 'gem':
            strategy = GEM(memory_strength = CONFIG.TRAIN.pretraining.memory_strength)
            self.callbacks.append(strategy)
        else:
            return 

    def setup(self) -> None:
        # self.prepare_FRN_HB_BWE_dataloaders()
        # self.prepare_FRN_MSM_clean_dataloaders()
        self.prepare_FRN_MSM_noisy_dataloaders()
        # self.prepare_FRN_MSM_aggressive_dataloaders()
        # self.prepare_FRN_NB_BWE_dataloaders()
        # self.prepare_FRN_NAE_dataloaders()
        self.setup_loggers()
        self.setup_model()
        # self.setup_base_strategies()
        self.setup_callbacks()
        self.setup_trainer()

    @abstractmethod
    def run_base_training(self) -> None:
        start = time.time()
        self.model.ema_mode = False
        self.model.ewc_mode = False
        self.model.regularizer_mode = False
        self.trainer.fit(model=self.model, train_dataloaders=self.train_dataloader, 
                         val_dataloaders=self.val_dataloader)
        self.baseline_experiment_time = time.time() - start
        
        self.trainer.save_checkpoint('baseline_training.ckpt')
        """Start training, validation, testing, and indexing loop here"""

    def execute(self) -> None:
            self.setup()
            self.run_base_training()
