import logging
import os
import random
from typing import Optional, List, Generator, Tuple

import hydra
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

# from continual_ranking.dpr.data.file_handler import read_json_file
# from continual_ranking.dpr.data.index_dataset import IndexDataset, IndexTokenizer
# from continual_ranking.dpr.data.train_dataset import TrainDataset, TrainTokenizer
from dataset.dataset_MSM import CustomDataset

logger = logging.getLogger(__name__)


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_sets = None
        self.eval_sets = None
        self.strategy = self.cfg.experiment.strategy

    def _read_training_data(self, is_train: bool) -> Tuple[List[dict], List[dict]]:
        if is_train:
            base_path = self.paths.train_base
            cl_path = self.paths.train_cl

        else:
            base_path = self.paths.val_base
            cl_path = self.paths.val_cl

        base_data = read_json_file(base_path)
        cl_data = read_json_file(cl_path)

        return base_data, cl_data

    @staticmethod
    def _make_baseline(base_data: list, cl_data: list, base_size: int, cl_sizes: List[int]) -> List[List[dict]]:
        base_set = base_data[:base_size]
        cl_set = cl_data[:max(cl_sizes)]

        data = base_set + cl_set
        random.shuffle(data)

        return [data]

    @staticmethod
    def _make_naive(base_data: list, cl_data: list, base_size: int, cl_sizes: List[int]) -> List[List[dict]]:
        base_set = base_data[:base_size]

        chunks = [base_set]
        cl_sizes = [0] + cl_sizes
        for i in range(len(cl_sizes) - 1):
            slice_ = cl_data[cl_sizes[i]: cl_sizes[i + 1]]
            chunks.append(slice_)

        return chunks

    def _make_replay(self, datasets: list, base_size: int, cl_sizes: List[int]) -> List[List[dict]]:
        logger.info('Preparing replay dataset')
        replays = [list(np.random.choice(chunk, int(len(chunk) * 0.2))) for chunk in datasets[:-1]]
        replays = [[], *replays]

        if self.strategy == 'replay':
            datasets = [dataset + replay for dataset, replay in zip(datasets, replays)]
        else:
            datasets = [chunk[len(replay):] + replay for chunk, replay in zip(datasets, replays)]

        for dataset in datasets[1:]:
            random.shuffle(dataset)

        return datasets

    def _make_set_splits(self, batch_size: int, split_size: float = 0) -> Generator[DataLoader, None, None]:
        base_size = self.cfg.experiment.base_size
        cl_sizes = list(self.cfg.experiment.cl_sizes)

        if split_size:
            base_size = int(base_size * split_size)
            cl_sizes = [int(size * split_size) for size in cl_sizes]

        base_data, cl_data = self._read_training_data(not bool(split_size))

        if self.strategy == 'baseline':
            logger.info('Preparing baseline dataset')
            datasets = self._make_baseline(base_data, cl_data, base_size, cl_sizes)

        else:
            datasets = self._make_naive(base_data, cl_data, base_size, cl_sizes)

            if self.strategy.startswith('replay'):
                datasets = self._make_replay(datasets, base_size, cl_sizes)

        for d in datasets:
            dataset = TrainDataset(d, self.cfg.negatives_amount, self.train_tokenizer)
            yield DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=self.cfg.biencoder.num_workers
            )

    def make_forgetting_dataset(self) -> DataLoader:
        base_data = read_json_file(self.paths.train_base)
        base_set = base_data[:self.cfg.experiment.base_size]
        dataset = TrainDataset(base_set, self.cfg.negatives_amount, self.train_tokenizer)
        return DataLoader(
            dataset,
            batch_size=self.cfg.biencoder.val_batch_size,
            num_workers=self.cfg.biencoder.num_workers
        )

    def setup(self, stage: Optional[str] = None):
        self.train_sets = self._make_set_splits(self.cfg.biencoder.train_batch_size)
        self.eval_sets = self._make_set_splits(self.cfg.biencoder.val_batch_size, self.cfg.datasets.split_size)

    def prepare_data(self) -> None:
        pass

    def get_ds(self, phase):
        ds = CustomDataset(phase)
        return ds

    
    def get_loader(self, phase):  
        ds = self.get_ds(phase)
        dl = DataLoader(ds, shuffle=False, batch_size=16,
                    num_workers=8, collate_fn=CustomDataset.collate_fn,
                    persistent_workers=True)
        return dl
    
    def train_dataloader(self):
        return self.get_loader(phase="train")

    def val_dataloader(self):
        return self.get_loader(phase="val")

    def test_dataloader(self):
        return self.get_loader(phase="test")
    

