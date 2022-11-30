import os, glob

from os import makedirs
from tqdm.auto import tqdm
from natsort import natsorted

from model.Tfilm import tfilm_net
from util import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_path = os.path.abspath(os.getcwd()) +  "/logs/model.h5"
 
lr_folder = "/lr_audio/"

save_folder = os.path.abspath(os.getcwd()) + '/output/model_name/'
makedirs(save_folder, exist_ok = True)

if __name__ == '__main__':

    model = tfilm_net()
    model.load_weights(model_path)
    model.summary()

    # generate SR audios from LR audios in 'lr_folder'
    if lr_folder is not None:
        paths = glob.glob(lr_folder + "*.wav")
        paths = natsorted(paths)

        names = os.listdir(lr_folder)
        names = natsorted(names)

        for i in tqdm(range(len(names))):
            generate_sr_sample(model, crop_length=8192,
                               in_dir_lr=paths[i], save_path=save_folder + names[i])