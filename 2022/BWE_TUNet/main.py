import torch

import argparse
import os, glob, time, random

import librosa
import numpy as np
import pytorch_lightning as pl
import soundfile as sf

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import CustomDataset
from dataset_sitec import CustomDataset_sitec
from models.tunet import TUNet
from models.tunet_realtime import TUNet_realtime
from models.tunet_realtime_atafilm import TUNet_reatime_atafilm
from utils.tblogger import TensorBoardLoggerExpanded
from utils.utils import evaluate_dataset, onnx_realtime_evaluate_dataset,frame, mkdir_p, overlap_add

# from config_folder.VCTK_TUNet import CONFIG
# from config_folder.VCTK_TUNet_realtime import CONFIG
from config_folder.VCTK_TUNet_realtime_atafilm import CONFIG
# from config_folder.SITEC_TUNet import CONFIG
# from config_folder.SITEC_TUNet_realtime import CONFIG
# from config_folder.SITEC_TUNet_realtime_atafilm import CONFIG


SAVE = True
ONNX = False
ONNX_ZERO = False
CPU = False
SINGLE = False


parser = argparse.ArgumentParser()

parser.add_argument('--version', default=None,
                    help='version to resume')
parser.add_argument('--mode', default='train',
                    help='training or testing mode')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG.gpus)
assert args.mode in ['train', 'eval', 'test', 'onnx'], "--mode should be 'train', 'eval', 'test' or 'onnx'"


def resume(train_dataset, val_dataset, version):
    print("Version", version)
    model_path = os.path.join(CONFIG.LOG.log_dir, 'version_{}/checkpoints/'.format(str(version)))
    config_path = os.path.join(CONFIG.LOG.log_dir, 'version_{}/'.format(str(version)) + 'hparams.yaml')
    model_name = [x for x in os.listdir(model_path) if x.endswith(".ckpt")][0]
    ckpt_path = model_path + model_name

    if CONFIG.MODEL.causal:
        checkpoint = TUNet_realtime.load_from_checkpoint(ckpt_path, hparams_file=config_path, train_dataset=train_dataset,
                                            val_dataset=val_dataset)
    elif CONFIG.MODEL.causal and CONFIG.MODEL.afilm:
        checkpoint = TUNet_reatime_atafilm.load_from_checkpoint(ckpt_path, hparams_file=config_path, train_dataset=train_dataset,
                                            val_dataset=val_dataset)
    else:
        checkpoint = TUNet.load_from_checkpoint(ckpt_path, hparams_file=config_path, train_dataset=train_dataset,
                                            val_dataset=val_dataset)
    return checkpoint


def train():
    if CONFIG.DATA.dataset == 'vctk':
        train_dataset = CustomDataset('train')
        val_dataset = CustomDataset('val')
    else:
        train_dataset = CustomDataset_sitec('train')
        val_dataset = CustomDataset_sitec('val')

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', verbose=True,
                                          filename='tunet-{epoch:02d}-{val_loss:.4f}', save_weights_only=False)
    gpus = CONFIG.gpus.split(',')
    logger = TensorBoardLoggerExpanded(CONFIG.DATA.sr)

    if args.version is not None:
        model = resume(train_dataset, val_dataset, args.version)
    else:
        if not CONFIG.MODEL.realtime:
            model = TUNet_realtime(train_dataset, val_dataset)
        elif CONFIG.MODEL.realtime and CONFIG.MODEL.atafilm:
            model = TUNet_reatime_atafilm(train_dataset, val_dataset)
        else:
            model = TUNet(train_dataset, val_dataset)
    
    print("Model is: ", model._get_name())
    print("Task is: ", CONFIG.TASK.task)
    
    trainer = pl.Trainer(logger=logger,
                         gradient_clip_val=CONFIG.TRAIN.clipping_val,
                         gpus=len(gpus),
                         max_epochs=CONFIG.TRAIN.epochs,
                         accelerator="ddp" if len(gpus) > 1 else None,
                         stochastic_weight_avg=True,
                         callbacks=[checkpoint_callback])

    print(model.hparams)
    print(
        'Dataset: {}, Train files: {}, Val files {}'.format(CONFIG.DATA.dataset, len(train_dataset), len(val_dataset)))
    trainer.fit(model)


def evaluate(model):
    sample_path = os.path.join(CONFIG.LOG.sample_path, "version_" + str(args.version))
    testset = CustomDataset('test')
    test_loader = DataLoader(testset, batch_size=1, num_workers=CONFIG.TRAIN.workers)
    version = args.version

    if not ONNX:
        res = evaluate_dataset(model, test_loader, sample_path, version, cpu = CPU, save = SAVE)
        print("evaluate without onnx")
        print("Version {} STOI: {} ESTOI: {} SNR:{} LSD: {} LSD-HF: {} SI-SDR: {}".format(args.version, res[0], res[1], res[2], res[3], res[4], res[5]))
        txt_name = './result/SITEC/version' + str(args.version) + '_' + CONFIG.TASK.task +'_result.txt' 
        file = open(txt_name, "w")
        f = "{0:<16} {1:<16}"
        file.write(f.format("Mean", "Std")+"\n")
        file.write("---------------------------------\n")
        file.write(str(res))
        file.write("\n")
        file.write("---------------------------------\n")
        metric = "STOI, ESTOI, SNR, LSD, LSD-HF, SI-SDR"
        file.write(metric)
        file.close()
        print(txt_name)
    
    else:
        print("evaluate with onnx real-time and zeropadding")
        res = onnx_realtime_evaluate_dataset(model, test_loader, sample_path, version, onnx_zero = ONNX_ZERO, save = SAVE)
        print("Version {} STOI: {} ESTOI: {} SNR: {} LSD: {} LSD-HF: {} SI-SDR: {}".format(args.version, res[0], res[1], res[2], res[3], res[4], res[5]))
        txt_name = './result/SITEC/version' + str(args.version) + '_' + CONFIG.TASK.task +'_zeropadding_16_result.txt' 
        file = open(txt_name, "w")
        f = "{0:<16} {1:<16}"
        file.write(f.format("Mean", "Std")+"\n")
        file.write("---------------------------------\n")
        file.write(str(res))
        file.write("\n")
        file.write("---------------------------------\n")
        metric = "STOI, ESTOI, SNR, LSD, LSD-HF, SI-SDR"
        file.write(metric)
        file.close()
        print(txt_name)
        
def test(model):
    in_dir = CONFIG.TEST.in_dir
    out_dir = CONFIG.TEST.out_dir
    
    files = glob.glob(in_dir +"*.wav")
    files.sort()
    print("all_files list is", len(files))

    mkdir_p(out_dir)
    window_size = CONFIG.DATA.window_size
    stride = CONFIG.DATA.stride
    latency = []
    n = 100
    
    for file in tqdm (random.sample(files, n)):
    # for file in tqdm(files):
        sig, sr = librosa.load(os.path.join(in_dir, file), sr=CONFIG.DATA.sr)
        d = max(len(sig) // stride + 1, 2) * stride
        sig = np.hstack((sig, np.zeros(d - len(sig))))
        x = frame(sig, window_size, stride)[:, np.newaxis, :]

        # x = torch.Tensor(x).cuda(device=0)
        # device = torch.device('cuda')

        device = torch.device('cpu')
        x = torch.Tensor(x).to(device)

        start = time.time()
        pred = model(x)
        latency.append(time.time() - start)
        
        pred = overlap_add(pred, window_size, stride, (1, 1, len(sig)))
        audio = np.squeeze(pred.detach().cpu().numpy())
        head, tail = os.path.split(file)

    
    print("PyTorch {} Inference time = {} ms".format(device.type, format(sum(latency) * 1000 / len(latency), '.2f')))
    print("max inference time ", max(latency))
    # latency.sort()
    # sorted(latency, reverse= True)

    if SINGLE:
        txt_name = './result/TIMIT/infrence_time_version' + str(args.version) + '_' + str(device.type) + '_thread1.txt'
    txt_name = './result/TIMIT/infrence_time_version' + str(args.version) + '_' + str(device.type) +'.txt'
    # txt_name = './result/TIMIT/inference_version8_torch_cpu_thread1.txt'
    line = "PyTorch {} Inference time = {} ms".format(device.type, format(sum(latency) * 1000 / len(latency), '.2f'))
    file = open(txt_name, "w")
    file.write(str(line))
    file.write("\n")
    file.close()
    print(txt_name)

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
        model.summarize()
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