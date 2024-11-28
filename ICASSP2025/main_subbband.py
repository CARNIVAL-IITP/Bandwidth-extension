import argparse
import os

import librosa
import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning.strategies.ddp import DDPStrategy

from config import CONFIG

from dataset import CustomDataset
from models.tunet import TUNet
from models.tunet_subband import TUNet_subband
from models.tunet_subband_pcl import TUNet_subband_pcl


from utils.tblogger import TensorBoardLoggerExpanded
from utils.utils import evaluate_dataset, frame, mkdir_p, overlap_add
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging as SWA
from pytorch_lightning.utilities.model_summary import summarize

from natsort import natsorted


SAVE = True
ONNX = False
ONNX_ZERO = False
CPU = False
SINGLE = False

parser = argparse.ArgumentParser()

parser.add_argument('--version', default=None,
                    help='version to resume')
parser.add_argument('--mode', default='train',
                    help='train, eval, ewc_train, onnx mode')
parser.add_argument('--subband', action='store_true',
                    help='turn on subband training mode')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.gpus)
assert args.mode in ['train', 'eval', 'test', 'onnx', 'finetune'], "--mode should be 'train', 'eval', 'test' or 'onnx'"


torch.autograd.set_detect_anomaly(True)

def resume(train_dataset, val_dataset, version):
    print("Version", version)
    # from config_pretrain import CONFIG
    model_path = os.path.join(CONFIG.LOG.log_dir, 'version_{}/checkpoints/'.format(str(version)))
    config_path = os.path.join(CONFIG.LOG.log_dir, 'version_{}/'.format(str(version)) + 'hparams.yaml')
    model_name = [x for x in os.listdir(model_path) if x.endswith(".ckpt")][0]
    ckpt_path = model_path + model_name

    if CONFIG.MODEL.model_name == 'TUNet-subband':
        checkpoint = TUNet_subband.load_from_checkpoint(ckpt_path, hparams_file=config_path, 
                                                    train_dataset=train_dataset,
                                            val_dataset=val_dataset, strict=False)
    elif CONFIG.MODEL.model_name == 'TUNet-subband-pcl':
        checkpoint = TUNet_subband_pcl.load_from_checkpoint(ckpt_path, hparams_file=config_path, 
                                                          train_dataset=train_dataset,
                                            val_dataset=val_dataset, strict=False)
    else:
        checkpoint = TUNet.load_from_checkpoint(ckpt_path, hparams_file=config_path, 
                                                train_dataset=train_dataset,
                                            val_dataset=val_dataset, strict= False)

    return checkpoint

def train():
    # from config_finetune import CONFIG
    train_dataset = CustomDataset('train')
    val_dataset = CustomDataset('val')

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', verbose=True,
                                          filename='tunet-{epoch:02d}-{val_loss:.4f}', 
                                          save_weights_only=False)
    gpus = CONFIG.gpus.split(',')
    logger = TensorBoardLoggerExpanded()

    if args.version is not None:
        model = resume(train_dataset, val_dataset, args.version)
    elif CONFIG.MODEL.model_name == 'TUNet-subband':
        model = TUNet_subband(train_dataset, val_dataset)
    elif CONFIG.MODEL.model_name == 'TUNet-subband-pcl':
        model = TUNet_subband_pcl(train_dataset, val_dataset)
    else:
        model = TUNet(train_dataset, val_dataset)

    trainer = pl.Trainer(logger=logger,
                        gradient_clip_val=CONFIG.TRAIN.clipping_val,
                        max_epochs=CONFIG.TRAIN.epochs,
                        accelerator="gpu",
                        devices=[0],
                        callbacks=[checkpoint_callback, lr_monitor])
                
    print(model.hparams)
    print(
        'Dataset: {}, Train files: {}, Val files {}'.format(CONFIG.DATA.dataset, 
                                                            len(train_dataset), 
                                                            len(val_dataset)))
    trainer.fit(model)



def evaluate(model):
    sample_path = os.path.join(CONFIG.LOG.sample_path, "version_" + str(args.version))

    testset = CustomDataset('test')
    test_loader = DataLoader(testset, batch_size=1, num_workers=CONFIG.TRAIN.workers)
    version = args.version

    res = evaluate_dataset(model, test_loader, sample_path, args.version, cpu = CPU, 
                           single = SINGLE, save = SAVE)
    print("Evaluate -- STOI: {} ESTOI: {} LSD_10: {} LSD: {} LSD-HF: {} LSD-LF: {} PESQ-WB: {} PESQ-NB: {} SNR:{} SI-SDR: {} newSI-SDR: {} SI-SIR: {} SI-SAR: {}".format( 
        res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], 
        res[9], res[10], res[11], res[12]))
    current_path = os.path.abspath(os.getcwd())
    result_path = current_path + '/result/' + CONFIG.DATA.dataset 
    os.makedirs(result_path, exist_ok = True)
    txt_name = result_path+'/'+CONFIG.DATA.dataset+'_'+CONFIG.MODEL.model_name+'_version'+str(args.version)+'_batch'+str(CONFIG.TRAIN.batch_size) +'_epoch' +str(CONFIG.TRAIN.epochs)+'_result.txt' 
    file = open(txt_name, "w")
    f = "{0:<16} {1:<16}"
    file.write(f.format("Mean", "Std")+"\n")
    file.write("---------------------------------\n")
    file.write(str(res))
    file.write("\n")
    file.write("---------------------------------\n")
    metric = "STOI, ESTOI, LSD_10, LSD, LSD-HF, LSD-LF, PESQ-WB, PESQ-NB, SNR, SI-SDR, newSI-SDR, SI-SIR, SI-SAR"
    file.write(metric)
    file.close()      


def test(model):
    in_dir = CONFIG.TEST.in_dir
    out_dir = CONFIG.TEST.out_dir
    files = os.listdir(in_dir)
    files = natsorted(files)
    mkdir_p(out_dir)
    window_size = CONFIG.DATA.window_size
    stride = CONFIG.DATA.stride

    for file in files:
        sig, sr = librosa.load(os.path.join(in_dir, file), sr=CONFIG.DATA.sr)
        d = max(len(sig) // stride + 1, 2) * stride
        sig = np.hstack((sig, np.zeros(d - len(sig))))
        x = frame(sig, window_size, stride)[:, np.newaxis, :]
        x = torch.Tensor(x).cuda(device=0)
        # pred = x
        pred = model(x)

        pred = overlap_add(pred, window_size, stride, (1, 1, len(sig)))
        audio = np.squeeze(pred.detach().cpu().numpy())
        sf.write(os.path.join(out_dir, file), audio, samplerate=sr, subtype='PCM_16')


def to_onnx(model, onnx_path):
    model.eval()
    x = torch.randn(1, 1, CONFIG.DATA.window_size)
    torch.onnx.export(model,
                      x,
                      onnx_path,
                      export_params=True,
                      opset_version=12,
                      input_names=['input'],
                      output_names=['output'],
                      do_constant_folding=True,
                      verbose=False)


if __name__ == '__main__':

    if args.mode == 'train':
        train()
    else:
        model = resume(None, None, args.version)
        print(model.hparams)
        summarize(model,max_depth=-1)
        model.eval()
        model.freeze()
        if args.mode == 'eval':
            model.cuda(device=0)
            evaluate(model)
        elif args.mode == 'test':
            model.cuda(device=0)
            test(model)
        else:
            onnx_path = 'lightning_logs/version_{}/checkpoints/tunet.onnx'.format(str(args.version))
            to_onnx(model, onnx_path)
            print('ONNX model saved to', onnx_path)
