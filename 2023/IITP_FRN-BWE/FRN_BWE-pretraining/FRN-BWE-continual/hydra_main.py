import logging
import logging.config
import os

import hydra
import torch.cuda
import yaml
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from continual_learning.continual_base import Experiment


def setup_logging() -> None:
    with open(os.path.join('config', 'logging.yaml'), 'r') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)


@hydra.main(config_path='../FRN-BWE-continual/config', config_name='base')
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info(f'\n{OmegaConf.to_yaml(cfg)}')

    seed_everything(42, workers=True)
    accelerator = 'gpu'

    cfg.device = accelerator

    baseline = Experiment(cfg)
    # baseline.execute()
    baseline.ctrl_execute()


if __name__ == "__main__":
    main()
