import os, h5py, librosa
import numpy as np

from tqdm.auto import tqdm


dimension = 8192                   # sequence length
stride = 4096                      # stride length when cutting the sequence

# the folder of HR audios and LR audios
in_dir_hr_train = "/path_to/train_hr/"
in_dir_lr_train = "/path_to/train_lr/"
in_dir_hr_test = "/path_to/test_hr/"
in_dir_lr_test = "/path_to/test_lr/"

# the path of output .h5 file
out_dir_train = "./train.h5"
out_dir_test = "./test.h5"


def add_data(in_dir_hr, in_dir_lr, out_dir):
    '''create training and testing set as .h5 files'''

    # Make a list of all files to be processed
    hr_files = os.listdir(in_dir_hr)
    hr_files.sort()
    hr_file_list = []
    for hr_file in hr_files:
        hr_file_list.append(in_dir_hr + hr_file)

    lr_files = os.listdir(in_dir_lr)
    lr_files.sort()
    lr_file_list = []
    for lr_file in lr_files:
        lr_file_list.append(in_dir_lr + lr_file)

    file_num = len(hr_file_list)
    assert file_num == len(lr_file_list)
  
    times = 0
    d = dimension
    s = stride
    for i in tqdm(range(file_num)):
        x_hr, fs_hr = librosa.load(hr_file_list[i], sr=None)
        x_lr, fs_lr = librosa.load(lr_file_list[i], sr=None)

        if len(x_hr) > len(x_lr):
            x_hr = x_hr[0:len(x_lr)]
            print("length error 1: ", i)
        elif len(x_hr) < len(x_lr):
            x_lr = x_lr[0:len(x_hr)]
            print("length error 2: ", i)

        assert fs_hr == fs_lr
        max_i = len(x_hr) - d + 1

        for j in range(0, max_i, s):

            hr_patch = np.array(x_hr[j: j + d])
            lr_patch = np.array(x_lr[j: j + d])

            assert len(hr_patch) == d
            assert len(lr_patch) == d

            if times == 0:
                f = h5py.File(out_dir, 'w')
                data_set = f.create_dataset(name='data', shape=(1, dimension, 1), dtype=np.float32,
                                            maxshape=(None, dimension, 1))
                label_set = f.create_dataset(name='label', shape=(1, dimension, 1), dtype=np.float32,
                                             maxshape=(None, dimension, 1))
            else:
                f = h5py.File(out_dir, 'a')
                data_set = f['data']
                label_set = f['label']

            data_set.resize([times + 1, dimension, 1])
            label_set.resize([times + 1, dimension, 1])
            data_set[times:times + 1] = lr_patch.reshape((1, d, 1))
            label_set[times:times + 1] = hr_patch.reshape((1, d, 1))
            times = times + 1
    f.close()

if __name__ == '__main__':
    print('\nCreating training dataset...')
    add_data(in_dir_hr=in_dir_hr_train, in_dir_lr=in_dir_lr_train, out_dir=out_dir_train)
    print('\nCreating testing dataset...')
    add_data(in_dir_hr=in_dir_hr_test, in_dir_lr=in_dir_lr_test, out_dir=out_dir_test)
