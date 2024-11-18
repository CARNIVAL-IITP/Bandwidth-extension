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
from models.frn_subband import FRN_Subband
from models.frn_subband_pcl import FRN_Subband_pcl
# from models.frn_encoder_only import PLCModel

from utils.tblogger import TensorBoardLoggerExpanded
from utils.utils import mkdir_p
from pytorch_lightning import seed_everything

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
    # if CONFIG.TRAIN.pretraining:
        # model_path = os.path.join('FRN_BWE-pretraining/', CONFIG.LOG.log_dir, 'version_{}/checkpoints/'.format(str(version)))
    # else: 
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
    elif CONFIG.MODEL.model_name == 'FRN-baseline-continual':
        checkpoint = ContinualFRN.load_from_checkpoint(ckpt_path,
                                                strict=False,
                                                train_dataset=train_dataset,
                                                val_dataset=val_dataset,
                                                version=version,
                                                save=CONFIG.TEST.save,
                                                window_size=CONFIG.DATA.window_size)
    elif CONFIG.MODEL.model_name == 'FRN-subband':
        checkpoint = FRN_Subband.load_from_checkpoint(ckpt_path,
                                                strict=True,
                                                train_dataset=train_dataset,
                                                val_dataset=val_dataset,
                                                version=version,
                                                save=CONFIG.TEST.save,
                                                window_size=CONFIG.DATA.window_size)
        
    elif CONFIG.MODEL.model_name == 'FRN-subband-pcl':
        checkpoint = FRN_Subband_pcl.load_from_checkpoint(ckpt_path,
                                                strict=True,
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


def train():
    train_dataset = TrainDataset('train')
    val_dataset = TrainDataset('val')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', verbose=True,
                                          filename='frn-{epoch:02d}-{val_loss:.4f}', save_weights_only=False)
    gpus = CONFIG.gpus.split(',')
    logger = TensorBoardLoggerExpanded(CONFIG.DATA.sr)

    if args.version is not None:
        model = resume(train_dataset, val_dataset, args.version)
    elif CONFIG.MODEL.model_name == 'FRN-baseline':
        model = PLCModel(train_dataset,
                         val_dataset)
    elif CONFIG.MODEL.model_name == 'FRN-FiLM':
        model = FRN_PLUS(train_dataset,
                        val_dataset)
    elif CONFIG.MODEL.model_name == 'FRN-encoder':
        model = PLCModel(train_dataset,
                        val_dataset)
    elif CONFIG.MODEL.model_name == 'FRN-baseline-continual':
        model = ContinualFRN(train_dataset,
                        val_dataset)
    elif CONFIG.MODEL.model_name == 'FRN-subband':
        model = FRN_Subband(train_dataset,
                        val_dataset)
    elif CONFIG.MODEL.model_name == 'FRN-subband-pcl':
        model = FRN_Subband_pcl(train_dataset,
                        val_dataset)
        
    trainer = pl.Trainer(accelerator="gpu", 
                         devices = [0],
                         logger=logger,
                         gradient_clip_val=CONFIG.TRAIN.clipping_val,
                         max_epochs=CONFIG.TRAIN.epochs,
                         callbacks=[checkpoint_callback]
                         )

    print(model.hparams)
    print(
        'Dataset: {}, Train files: {}, Val files {}'.format(CONFIG.DATA.dataset, len(train_dataset), len(val_dataset)))
    trainer.fit(model)


def to_onnx(model, onnx_path):
    model.eval()

    model = OnnxWrapper(model)

    torch.onnx.export(model,
                      model.sample,
                      onnx_path,
                      export_params=True,
                      opset_version=12,
                      input_names=model.input_names,
                      output_names=model.output_names,
                      do_constant_folding=True,
                      verbose=False)


if __name__ == '__main__':
    seed_everything(42, workers=True)
    if args.mode == 'train':
        train()
    else:
        model = resume(None, None, args.version)
        print(model.hparams)
        print(summarize(model))

        model.eval()
        model.freeze()
        if args.mode == 'eval':
            model.cuda(device=0)
            trainer = pl.Trainer(accelerator='gpu', devices=1, enable_checkpointing=False, logger=False)
            testset = TestLoader()
            test_loader = DataLoader(testset, batch_size=1, num_workers=8)
            metrics = trainer.test(model, test_loader)
            # print(inference_time)
            # exit()
            current_path = os.path.abspath(os.getcwd())
            result_path = current_path + '/result/' + CONFIG.DATA.dataset + '/'
            os.makedirs(result_path, exist_ok = True)
            txt_name = result_path + CONFIG.DATA.dataset +'_'+CONFIG.MODEL.model_name+'_linux1_version' + str(args.version) + '_' + CONFIG.TASK.task +'_result.txt'             
            file = open(txt_name, "w")
            file.write(str(metrics))
            exit()
            print('Version', args.version)
            masking = CONFIG.DATA.EVAL.masking
            prob = CONFIG.DATA.EVAL.transition_probs[0]
            loss_percent = (1 - prob[0]) / (2 - prob[0] - prob[1]) * 100
            print('Evaluate with real trace' if masking == 'real' else
                  'Evaluate with generated trace with {:.2f}% packet loss'.format(loss_percent))
        elif args.mode == 'test':
            model.cuda(device=0)
            testset = BlindTestLoader(test_dir=CONFIG.TEST.in_dir)
            test_loader = DataLoader(testset, batch_size=1, num_workers=8)
            trainer = pl.Trainer(accelerator='gpu', devices=1, enable_checkpointing=False, logger=False)
            preds = trainer.predict(model, test_loader, return_predictions=True)
            mkdir_p(CONFIG.TEST.out_dir)
            for idx, path in enumerate(test_loader.dataset.data_list):
                out_path = os.path.join(CONFIG.TEST.out_dir, os.path.basename(path))
                sf.write(out_path, preds[idx], samplerate=CONFIG.DATA.sr, subtype='PCM_16')

        else:
            onnx_path = 'lightning_logs/version_{}/checkpoints/frn.onnx'.format(str(args.version))
            to_onnx(model, onnx_path)
            print('ONNX model saved to', onnx_path)
