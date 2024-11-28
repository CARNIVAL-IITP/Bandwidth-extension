# Bandwidth extension (BWE)
This code is for "Subband Modeling for Artifacts-free Speech Super-resolution" which is subbmitted to ICASSP 2025.

We have provided the traning code and inference code for reproducibility.
In addition, we have provided demo samples under "samples" folder.

## 1. Setup 
Please refer to the orignal code link above

### Requirements
* PyTorch
* Numpy
* h5py

To install all required packages via pip command
```
Pip3 install -r tunet_requirments.txt
```

## 2. Prepare dataset

English Speech TIMIT & VCTK DB 사용

* Download and extract the datasets for VCTK:
    ```
    $ wget http://www.udialogue.org/download/VCTK-Corpus.tar.gz -O data/vctk/VCTK-Corpus.tar.gz
    $ tar -zxvf data/vctk/VCTK-Corpus.tar.gz -C data/vctk/ --strip-components=1
    ```

  After extracting the datasets, your `./data` directory should look like this:

    ```
    .
    |--data
        |--vctk
            |--wav48
                |--p225
                    |--p225_001.wav
                    ...
            |--train.txt   
            |--test.txt
        |--TIMIT
            |--train
                |--SA1_5.wav
                        ...                
            |--test
                |--SA1_2.wav
                        ...      
            |--train.txt   
            |--test.txt
    ```
* In order to load the datasets, text files that contain training and testing audio paths are required.

## 3. Training

### Configuration
Modify parameter settings in config.py to select "TUNet-baseline, TUNet-subband, TUNet-subband-pcl" 

### To train

* Run `main_subband.py`:
    ```
    $ python main_subband.py --mode train
    ```

## 4. Test

### Audio generation & Evaluation
* Run `main_subband.py` with a version number to be evaluated:
    ```
    $ python main_subband.py --mode eval --version 5
    ```

## 5. Reference
* TUNet paper: https://ieeexplore.ieee.org/abstract/document/9747699
