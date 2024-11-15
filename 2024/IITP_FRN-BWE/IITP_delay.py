import argparse
import os

import pytorch_lightning as pl
import soundfile as sf
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import summarize
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import TrainDataset, TestLoader, BlindTestLoader

from models.frn import PLCModel, OnnxWrapper
from models.ftn_film import FRN_PLUS
from models.frn_continual import ContinualFRN
# from models.frn_encoder_only import PLCModel

from utils.tblogger import TensorBoardLoggerExpanded
from utils.utils import mkdir_p
from pytorch_lightning import seed_everything


def resume(train_dataset, val_dataset, version):
    print("Version", version)
    if CONFIG.TRAIN.pretraining:
        model_path = os.path.join('FRN_BWE-pretraining/', CONFIG.LOG.log_dir, 'version_{}/checkpoints/'.format(str(version)))
    else: 
        model_path = os.path.join(CONFIG.LOG.log_dir, 'version_{}/checkpoints/'.format(str(version)))
    hparams_path = os.path.join(CONFIG.LOG.log_dir, 'version_{}/'.format(str(version)) + 'hparams.yaml')
    model_name = [x for x in os.listdir(model_path) if x.endswith(".ckpt")][0]
    ckpt_path = model_path + model_name

    if CONFIG.MODEL.model_name == 'FRN-baseline':
        checkpoint = PLCModel.load_from_checkpoint(ckpt_path,
                                               strict=False,
                                               train_dataset=train_dataset,
                                               val_dataset=val_dataset,
                                               version=version,
                                               save=CONFIG.TEST.save,
                                               window_size=CONFIG.DATA.window_size)
    elif CONFIG.MODEL.model_name == 'FRN-FiLM':
        checkpoint = FRN_PLUS.load_from_checkpoint(ckpt_path,
                                                strict=True,
                                                train_dataset=train_dataset,
                                                val_dataset=val_dataset,
                                                version=version,
                                                save=CONFIG.TEST.save,
                                                window_size=CONFIG.DATA.window_size)
    elif CONFIG.MODEL.model_name == 'FRN-continual-baseline':
        checkpoint = ContinualFRN.load_from_checkpoint(ckpt_path,
                                                strict=False,
                                                train_dataset=train_dataset,
                                                val_dataset=val_dataset,
                                                version=version,
                                                save=CONFIG.TEST.save,
                                                window_size=CONFIG.DATA.window_size)
    else:
        checkpoint = PLCModel.load_from_checkpoint(ckpt_path,
                                               strict=True,
                                               train_dataset=train_dataset,
                                               val_dataset=val_dataset,
                                               version=version,
                                               save=CONFIG.TEST.save,
                                               window_size=CONFIG.DATA.window_size)
    return checkpoint


if __name__ == '__main__':
    version = 17
    model = resume(None, None, 17)
    model.eval()
    model.freeze()

    model.cuda(device=0)
    testset = BlindTestLoader(test_dir=CONFIG.TEST.in_dir)
    test_loader = DataLoader(testset, batch_size=1, num_workers=8)
    trainer = pl.Trainer(accelerator='gpu', devices=1, enable_checkpointing=False, logger=False)
    preds = trainer.predict(model, test_loader, return_predictions=True)
    mkdir_p(CONFIG.TEST.out_dir)
    for idx, path in enumerate(test_loader.dataset.data_list):
        out_path = os.path.join(CONFIG.TEST.out_dir, os.path.basename(path))
        sf.write(out_path, preds[idx], samplerate=CONFIG.DATA.sr, subtype='PCM_16')

