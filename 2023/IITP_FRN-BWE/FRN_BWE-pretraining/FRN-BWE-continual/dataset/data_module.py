import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset.dataset_FRN_HB_BWE import TrainDataset_FRN_HB_BWE
from dataset.dataset_FRN_MSM_clean import TrainDataset_FRN_MSM_clean
from dataset.dataset_FRN_MSM_noisy import TrainDataset_FRN_MSM_noisy
from dataset.dataset_FRN_MSM_aggressive import TrainDataset_FRN_MSM_aggressive
from dataset.dataset_FRN_NB_BWE import TrainDataset_FRN_NB_BWE
from dataset.dataset_FRN_NAE import TrainDataset_FRN_NAE

from dataset.dataset_MSM import CustomDataset_MSM
from dataset.dataset_MSM_clean import CustomDataset_MSM_clean
from dataset.dataset_MSM_noisy import CustomDataset_MSM_noisy
from dataset.dataset_MSM_aggressive import CustomDataset_MSM_aggressive
from dataset.dataset_NB_BWE import CustomDataset_NB_BWE
from dataset.dataset_NAE import CustomDataset_NAE
from dataset.dataset_HB_BWE import CustomDataset_HB_BWE


logger = logging.getLogger(__name__)


class DataModule_FRN_HB_BWE(pl.LightningDataModule):
    def __init__(self):
        super(DataModule_FRN_HB_BWE, self).__init__()

    def prepare_data(self) -> None:
        pass

    def get_ds(self, phase):
        ds = TrainDataset_FRN_HB_BWE(phase)
        return ds

    def train_dataloader(self):
        ds = self.get_ds(phase='train')
        dl = DataLoader(ds, shuffle=True, batch_size=16,
                        num_workers=8, 
                        pin_memory=True, persistent_workers=True)
        return dl

    def val_dataloader(self):
        ds = self.get_ds(phase='val')
        dl = DataLoader(ds, shuffle=False, batch_size=16,
                        num_workers=8, 
                        pin_memory=True, persistent_workers=True)
        return dl
    
    def test_dataloader(self):
        ds = self.get_ds(phase='test')
        dl = DataLoader(ds, shuffle=False, batch_size=1,
                        num_workers=8, 
                        pin_memory=True, persistent_workers=True)
        

class DataModule_FRN_NB_BWE(pl.LightningDataModule):
    def __init__(self):
        super(DataModule_FRN_NB_BWE, self).__init__()

    def prepare_data(self) -> None:
        pass

    def get_ds(self, phase):
        ds = TrainDataset_FRN_NB_BWE(phase)
        return ds

    def train_dataloader(self):
        ds = self.get_ds(phase='train')
        dl = DataLoader(ds, shuffle=True, batch_size=16,
                        num_workers=8, 
                        pin_memory=True, persistent_workers=True)
        return dl

    def val_dataloader(self):
        ds = self.get_ds(phase='val')
        dl = DataLoader(ds, shuffle=False, batch_size=16,
                        num_workers=8, 
                        pin_memory=True, persistent_workers=True)
        return dl
    
    def test_dataloader(self):
        ds = self.get_ds(phase='test')
        dl = DataLoader(ds, shuffle=False, batch_size=1,
                        num_workers=8, 
                        pin_memory=True, persistent_workers=True)
        
class DataModule_FRN_NAE(pl.LightningDataModule):
    def __init__(self):
        super(DataModule_FRN_NAE, self).__init__()

    def prepare_data(self) -> None:
        pass

    def get_ds(self, phase):
        ds = TrainDataset_FRN_NAE(phase)
        return ds

    def train_dataloader(self):
        ds = self.get_ds(phase='train')
        dl = DataLoader(ds, shuffle=True, batch_size=16,
                        num_workers=8, 
                        pin_memory=True, persistent_workers=True)
        return dl

    def val_dataloader(self):
        ds = self.get_ds(phase='val')
        dl = DataLoader(ds, shuffle=False, batch_size=16,
                        num_workers=8, 
                        pin_memory=True, persistent_workers=True)
        return dl
    
    def test_dataloader(self):
        ds = self.get_ds(phase='test')
        dl = DataLoader(ds, shuffle=False, batch_size=1,
                        num_workers=8, 
                        pin_memory=True, persistent_workers=True)
        
class DataModule_FRN_MSM_clean(pl.LightningDataModule):
    def __init__(self):
        super(DataModule_FRN_MSM_clean, self).__init__()

    def prepare_data(self) -> None:
        pass

    def get_ds(self, phase):
        ds = TrainDataset_FRN_MSM_clean(phase)
        return ds

    def train_dataloader(self):
        ds = self.get_ds(phase='train')
        dl = DataLoader(ds, shuffle=True, batch_size=16,
                        num_workers=8, 
                        pin_memory=True, persistent_workers=True)
        return dl

    def val_dataloader(self):
        ds = self.get_ds(phase='val')
        dl = DataLoader(ds, shuffle=False, batch_size=16,
                        num_workers=8, 
                        pin_memory=True, persistent_workers=True)
        return dl
    
    def test_dataloader(self):
        ds = self.get_ds(phase='test')
        dl = DataLoader(ds, shuffle=False, batch_size=1,
                        num_workers=8, 
                        pin_memory=True, persistent_workers=True)

class DataModule_FRN_MSM_noisy(pl.LightningDataModule):
    def __init__(self):
        super(DataModule_FRN_MSM_noisy, self).__init__()

    def prepare_data(self) -> None:
        pass

    def get_ds(self, phase):
        ds = TrainDataset_FRN_MSM_noisy(phase)
        return ds

    def train_dataloader(self):
        ds = self.get_ds(phase='train')
        dl = DataLoader(ds, shuffle=True, batch_size=16,
                        num_workers=8, 
                        pin_memory=True, persistent_workers=True)
        return dl

    def val_dataloader(self):
        ds = self.get_ds(phase='val')
        dl = DataLoader(ds, shuffle=False, batch_size=16,
                        num_workers=8, 
                        pin_memory=True, persistent_workers=True)
        return dl
    
    def test_dataloader(self):
        ds = self.get_ds(phase='test')
        dl = DataLoader(ds, shuffle=False, batch_size=1,
                        num_workers=8, 
                        pin_memory=True, persistent_workers=True)

class DataModule_FRN_MSM_aggressive(pl.LightningDataModule):
    def __init__(self):
        super(DataModule_FRN_MSM_aggressive, self).__init__()

    def prepare_data(self) -> None:
        pass

    def get_ds(self, phase):
        ds = TrainDataset_FRN_MSM_aggressive(phase)
        return ds

    def train_dataloader(self):
        ds = self.get_ds(phase='train')
        dl = DataLoader(ds, shuffle=True, batch_size=16,
                        num_workers=8, 
                        pin_memory=True, persistent_workers=True)
        return dl

    def val_dataloader(self):
        ds = self.get_ds(phase='val')
        dl = DataLoader(ds, shuffle=False, batch_size=16,
                        num_workers=8, 
                        pin_memory=True, persistent_workers=True)
        return dl
    
    def test_dataloader(self):
        ds = self.get_ds(phase='test')
        dl = DataLoader(ds, shuffle=False, batch_size=1,
                        num_workers=8, 
                        pin_memory=True, persistent_workers=True)
        
class DataModule_MSM_clean(pl.LightningDataModule):
    def __init__(self):
        super(DataModule_MSM_clean, self).__init__()

    def prepare_data(self) -> None:
        pass

    def get_ds(self, phase):
        ds = CustomDataset_MSM_clean(phase)
        return ds

    def train_dataloader(self):
        ds = self.get_ds(phase='train')
        dl = DataLoader(ds, shuffle=True, batch_size=16,
                        num_workers=8, collate_fn=CustomDataset_MSM_clean.collate_fn,
                        pin_memory=True, persistent_workers=True)
        return dl

    def val_dataloader(self):
        ds = self.get_ds(phase='val')
        dl = DataLoader(ds, shuffle=False, batch_size=16,
                        num_workers=8, collate_fn=CustomDataset_MSM_clean.collate_fn,
                        pin_memory=True, persistent_workers=True)
        return dl
    
    def test_dataloader(self):
        ds = self.get_ds(phase='test')
        dl = DataLoader(ds, shuffle=False, batch_size=1,
                        num_workers=8, collate_fn=CustomDataset_MSM_clean.collate_fn,
                        pin_memory=True, persistent_workers=True)
        

class DataModule_MSM_noisy(pl.LightningDataModule):
    def __init__(self):
        super(DataModule_MSM_noisy, self).__init__()

    def prepare_data(self) -> None:
        pass

    def get_ds(self, phase):
        ds = CustomDataset_MSM_noisy(phase)
        return ds

    def train_dataloader(self):
        ds = self.get_ds(phase='train')
        dl = DataLoader(ds, shuffle=True, batch_size=16,
                        num_workers=8, collate_fn=CustomDataset_MSM_noisy.collate_fn,
                        pin_memory=True, persistent_workers=True)
        return dl

    def val_dataloader(self):
        ds = self.get_ds(phase='val')
        dl = DataLoader(ds, shuffle=False, batch_size=16,
                        num_workers=8, collate_fn=CustomDataset_MSM_noisy.collate_fn,
                        pin_memory=True, persistent_workers=True)
        return dl
    
    def test_dataloader(self):
        ds = self.get_ds(phase='test')
        dl = DataLoader(ds, shuffle=False, batch_size=1,
                        num_workers=8, collate_fn=CustomDataset_MSM_noisy.collate_fn,
                        pin_memory=True, persistent_workers=True)
        return dl
    
class DataModule_MSM_aggressive(pl.LightningDataModule):
    def __init__(self):
        super(DataModule_MSM_aggressive, self).__init__()

    def prepare_data(self) -> None:
        pass

    def get_ds(self, phase):
        ds = CustomDataset_MSM_aggressive(phase)
        return ds

    def train_dataloader(self):
        ds = self.get_ds(phase='train')
        dl = DataLoader(ds, shuffle=True, batch_size=16,
                        num_workers=8, collate_fn=CustomDataset_MSM_aggressive.collate_fn,
                        pin_memory=True, persistent_workers=True)
        return dl

    def val_dataloader(self):
        ds = self.get_ds(phase='val')
        dl = DataLoader(ds, shuffle=False, batch_size=16,
                        num_workers=8, collate_fn=CustomDataset_MSM_aggressive.collate_fn,
                        pin_memory=True, persistent_workers=True)
        return dl
    
    def test_dataloader(self):
        ds = self.get_ds(phase='test')
        dl = DataLoader(ds, shuffle=False, batch_size=1,
                        num_workers=8, collate_fn=CustomDataset_MSM_aggressive.collate_fn,
                        pin_memory=True, persistent_workers=True)
        return dl
    
class DataModule_MSM(pl.LightningDataModule):
    def __init__(self):
        super(DataModule_MSM, self).__init__()

    def prepare_data(self) -> None:
        pass

    def get_ds(self, phase):
        ds = CustomDataset_MSM(phase)
        return ds

    def train_dataloader(self):
        ds = self.get_ds(phase='train')
        dl = DataLoader(ds, shuffle=True, batch_size=16,
                        num_workers=8, collate_fn=CustomDataset_MSM.collate_fn,
                        pin_memory=True, persistent_workers=True)
        return dl

    def val_dataloader(self):
        ds = self.get_ds(phase='val')
        dl = DataLoader(ds, shuffle=False, batch_size=16,
                        num_workers=8, collate_fn=CustomDataset_MSM.collate_fn,
                        pin_memory=True, persistent_workers=True)
        return dl
    
    def test_dataloader(self):
        ds = self.get_ds(phase='test')
        dl = DataLoader(ds, shuffle=False, batch_size=1,
                        num_workers=8, collate_fn=CustomDataset_MSM.collate_fn,
                        pin_memory=True, persistent_workers=True)
        return dl
    

class DataModule_NB_BWE(pl.LightningDataModule):
    def __init__(self):
        super(DataModule_NB_BWE, self).__init__()

    def prepare_data(self) -> None:
        pass

    def get_ds(self, phase):
        ds = CustomDataset_NB_BWE(phase)
        return ds

    def train_dataloader(self):
        ds = self.get_ds(phase='train')
        dl = DataLoader(ds, shuffle=True, batch_size=16,
                        num_workers=8, collate_fn=CustomDataset_NB_BWE.collate_fn,
                        pin_memory=True, persistent_workers=True)
        return dl

    def val_dataloader(self):
        ds = self.get_ds(phase='val')
        dl = DataLoader(ds, shuffle=False, batch_size=16,
                        num_workers=8, collate_fn=CustomDataset_NB_BWE.collate_fn,
                        pin_memory=True, persistent_workers=True)
        return dl
    
    def test_dataloader(self):
        ds = self.get_ds(phase='test')
        dl = DataLoader(ds, shuffle=False, batch_size=1,
                        num_workers=8, collate_fn=CustomDataset_NB_BWE.collate_fn,
                        pin_memory=True, persistent_workers=True)
        return dl

class DataModule_NAE(pl.LightningDataModule):
    def __init__(self):
        super(DataModule_NAE, self).__init__()

    def prepare_data(self) -> None:
        pass

    def get_ds(self, phase):
        ds = CustomDataset_NAE(phase)
        return ds

    def train_dataloader(self):
        ds = self.get_ds(phase='train')
        dl = DataLoader(ds, shuffle=True, batch_size=16,
                        num_workers=8, collate_fn=CustomDataset_NAE.collate_fn,
                        pin_memory=True, persistent_workers=True)
        return dl

    def val_dataloader(self):
        ds = self.get_ds(phase='val')
        dl = DataLoader(ds, shuffle=False, batch_size=16,
                        num_workers=8, collate_fn=CustomDataset_NAE.collate_fn,
                        pin_memory=True, persistent_workers=True)
        return dl
    
    def test_dataloader(self):
        ds = self.get_ds(phase='test')
        dl = DataLoader(ds, shuffle=False, batch_size=1,
                        num_workers=8, collate_fn=CustomDataset_NAE.collate_fn,
                        pin_memory=True, persistent_workers=True)
        return dl
    
class DataModule_HB_BWE(pl.LightningDataModule):
    def __init__(self):
        super(DataModule_HB_BWE, self).__init__()

    def prepare_data(self) -> None:
        pass

    def get_ds(self, phase):
        ds = CustomDataset_HB_BWE(phase)
        return ds

    def train_dataloader(self):
        ds = self.get_ds(phase='train')
        dl = DataLoader(ds, shuffle=True, batch_size=16,
                        num_workers=8, collate_fn=CustomDataset_HB_BWE.collate_fn,
                        pin_memory=True, persistent_workers=True)
        return dl

    def val_dataloader(self):
        ds = self.get_ds(phase='val')
        dl = DataLoader(ds, shuffle=False, batch_size=16,
                        num_workers=8, collate_fn=CustomDataset_HB_BWE.collate_fn,
                        pin_memory=True, persistent_workers=True)
        return dl
    
    def test_dataloader(self):
        ds = self.get_ds(phase='test')
        dl = DataLoader(ds, shuffle=False, batch_size=1,
                        num_workers=8, collate_fn=CustomDataset_HB_BWE.collate_fn,
                        pin_memory=True, persistent_workers=True)
        return dl
    