import logging
import time
from typing import Optional

import torch
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from pytorch_lightning.strategies.ddp import DDPStrategy

from utils.hydra_utls import pickle_dump
# from continual_ranking.dpr.evaluator import Evaluator
from continual_learning.base import Base
# import wandb
from continual_learning.strategy.ewc import EWC
from continual_learning.strategy.gem import GEM
from typing import Optional, List

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


# from dataset.original_data_module import DataModule
from dataset.data_module import DataModule_HB_BWE, DataModule_NAE
from pytorch_lightning import Callback
from utils.hydra_utls import EMA
from pynvml import *
from continual_learning.trainer import ContinualTrainer
from utils.tblogger import TensorBoardLoggerExpanded

from config_folder.config_FRN_HB_BWE import CONFIG
from pytorch_lightning import Trainer
from continual_learning.strategy.ewc import EWC

logger = logging.getLogger(__name__)


class Experiment(Base):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.experiment_id: int = 0
        self.training_time: float = 0
        self.test_path: str = ''
        self.index_path: str = ''
        self.forgetting_dataloader: Optional[DataLoader] = None
        self.ewc: Optional[EWC] = None

    def wandb_log(self, metrics: dict):
        if self.logging_on:
            wandb.log(metrics)

    def setup_ctrl_strategies(self) -> None:
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


    def setup_ctrl_NAE_trainer(self) -> None:
        logger = TensorBoardLoggerExpanded()
        self.trainer = ContinualTrainer(
            gradient_clip_val = self.cfg.clipping_val,
            max_epochs = 100, 
            accelerator = "gpu",
            # devices = [0],
            devices = [1],
            callbacks=self.callbacks,
        )

    def setup_ctrl_HB_BWE_trainer(self) -> None:
        logger = TensorBoardLoggerExpanded()
        self.trainer = ContinualTrainer(
            gradient_clip_val = self.cfg.clipping_val,
            max_epochs = CONFIG.TRAIN.epochs, 
            accelerator = "gpu",
            # strategy = DDPStrategy(find_unused_parameters = False),
            # devices = [0, 1],
            # devices = [0],
            devices = [1],
            logger = logger,
            callbacks = self.callbacks,
        )

    def prepare_ctrl_FRN_NAE_dataloaders(self) -> None:
        logger.info('Setting up dataloaders')
        self.datamodule = DataModule_FRN_NAE()
        self.train_dataloader = self.datamodule.train_dataloader()
        self.val_dataloader = self.datamodule.val_dataloader()
        self.train_dataset = TrainDataset_FRN_NAE('train')
        self.val_dataset = TrainDataset_FRN_NAE('val')
        self.train_length = len(self.train_dataset)
        self.val_length = len(self.val_dataset)
        self.experiment_name = 'NAE'


    def prepare_ctrl_FRN_HB_BWE_dataloaders(self) -> None:
        logger.info('Setting up dataloaders')
        self.datamodule = DataModule_FRN_HB_BWE()
        self.train_dataloader = self.datamodule.train_dataloader()
        self.val_dataloader = self.datamodule.val_dataloader()
        self.train_dataset = TrainDataset_FRN_HB_BWE('train')
        self.val_dataset = TrainDataset_FRN_HB_BWE('val')
        self.train_length = len(self.train_dataset)
        self.val_length = len(self.val_dataset)
        self.experiment_name = 'HB-BWE'

    def resume_from_checkpoint(self):
        version = None
        # model_path = os.path.join(CONFIG.LOG.log_dir, 'version_{}/checkpoints/'.format(str(version)))
        model_path = os.path.join(CONFIG.LOG.log_dir, 'checkpoints/')
        model_name = [x for x in os.listdir(model_path) if x.endswith(".ckpt")][0]
        ckpt_path = model_path + model_name
        if CONFIG.MODEL.model_name == 'FRN-baseline':
            checkpoint = ContinualFRN.load_from_checkpoint(ckpt_path,
                                                strict=False,
                                                train_dataset=self.train_dataset,
                                                val_dataset=self.val_dataset,
                                                version=version,
                                                save=CONFIG.TEST.save,
                                                window_size=CONFIG.DATA.window_size)
        elif CONFIG.MODEL.model_name == 'FRN-FiLM':
            checkpoint = ContinualFRN_film.load_from_checkpoint(ckpt_path,
                                                strict=False,
                                                train_dataset=self.train_dataset,
                                                val_dataset=self.val_dataset,
                                                version=version,
                                                save=CONFIG.TEST.save,
                                                window_size=CONFIG.DATA.window_size)
        return checkpoint

    def setup_ctrl_model(self) -> None:
        logger.info('Setting up model')
        self.model = self.resume_from_checkpoint()
        print(self.model.hparams)

    def run_ctrl_training(self) -> None:
        start = time.time()
        if CONFIG.TRAIN.pretraining.strategy == 'ewc':
            self._continual_strategies(self.train_dataloader)
        self.trainer.fit(model=self.model, train_dataloaders=self.train_dataloader, 
                         val_dataloaders=self.val_dataloader)
        experiment_time = time.time() - start
    
        # print("baseline training time: ", self.baseline_experiment_time)
        print("ctrl training time: ", experiment_time)

        self.trainer.save_checkpoint(f'{self.experiment_name}.ckpt')

    def _continual_strategies(self, train_dataloader: DataLoader):
        # if self.ewc and self.trainer.task_id < self.trainer.tasks:
        self.ewc.train_dataloader = train_dataloader
        self.ewc.calculate_importances(self.trainer, self.model, train_dataloader)

    def ctrl_NAE_setup(self) -> None:
        self.prepare_ctrl_FRN_NAE_dataloaders()
        # self.setup_loggers()
        # self.setup_model()
        # self.setup_strategies()
        # self.setup_ctrl_callbacks()
        self.setup_ctrl_NAE_trainer()

    def ctrl_HB_BWE_setup(self) -> None:
        self.prepare_ctrl_FRN_HB_BWE_dataloaders()
        # self.setup_model()
        self.setup_ctrl_model()
        self.setup_loggers()
        self.setup_callbacks()
        self.setup_ctrl_strategies()
        self.setup_ctrl_HB_BWE_trainer()

    def ctrl_execute(self) -> None:
        # self.ctrl_NAE_setup()
        # self.run_ctrl_training()
        self.ctrl_HB_BWE_setup()
        self.run_ctrl_training()
