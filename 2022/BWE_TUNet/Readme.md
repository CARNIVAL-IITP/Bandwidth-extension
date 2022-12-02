# Bandwidth extension (BWE)
본 코드는 2021년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한 "원격다자간 영상회의에서의 음성 품질 고도화 기술개발" 과제의 일환으로 공개된 명료도 향상 부문의 2차년도 코드입니다. 

본 코드의 특징은 다음과 같습니다.
* This code is from https://github.com/NXTProduct/TUNet
* 음성 명료도 저하 극복을 위한 대역폭 확장 BWE 알고리즘 딥러닝 모델 TUNet을 baseline으로 선정
* 본 과제의 목적인 실시간성에 맞춰서 기존의 convolution을 causal convolution으로 대체하여 TUNet_realtime으로 모델 변경
* Attention mechanism을 사용하는 AFiLM을 추가하여 TUNet_realtime_ATAFiLM 모델 구조 고도화 
* 사전학습 Pretraining을 통하여 성능향상 도모

## Setup 
자세한 환경 설치 방법은 위 링크 참고

### Requirements
* PyTorch
* Numpy
* h5py

To install all required packages via pip command
```
Pip3 install -r tunet_requirments.txt
```

## Prpare dataset

영어 음성 VCTK DB 및 한굿어 음성 SITEC DB 사용
SITEC DB는 다음과 같은 room환경에서 RIR을 통하여 데이터셋 생성

* Download and extract the datasets for VCTK:
    ```
    $ wget http://www.udialogue.org/download/VCTK-Corpus.tar.gz -O data/vctk/VCTK-Corpus.tar.gz
    $ wget https://ailab.hcmus.edu.vn/assets/vivos.tar.gz -O data/vivos/vivos.tar.gz
    $ tar -zxvf data/vctk/VCTK-Corpus.tar.gz -C data/vctk/ --strip-components=1
    $ tar -zxvf data/vivos/vivos.tar.gz -C data/vivos/ --strip-components=1
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
        |--sitec
            |--train
                |--waves
                    |--111
                        |--1110000019272_v_111.wav
                        ...                
            |--val
                |--waves
                    |--111
                        |--1110000019272_v_111.wav
                        ...      
            |--test
                |--waves
                    |--111
                        |--1110000019272_v_111.wav
                        ...      
            |--sitec_rir_each_train.txt   
            |--sitec_rir_each_test.txt
            |--sitec_rir_each_val.txt
    ```
* In order to load the datasets, text files that contain training and testing audio paths are required. We have
  prepared `train.txt` and `test.txt` files in `./data/vctk` and `./data/sitec` directories.

## Training

### Configuration
config_foler 폴더에 있는 참조
* VCTK DB에 TUNet baseline: `VCTK_TUNet.py`
* VCTK DB에 TUNet_realtime: `VCTK_TUNet_realtime.py`
* VCTK DB에 TUNet_realtime 모델에 atafilm 적용: `VCTK_TUNet_realtime_atafilm.py`
* SITEC DB에 TUNet baseline: `SITEC_TUNet.py`
* SITEC DB에 TUNet_realtime: `SITEC_TUNet_realtime.py`
* SITEC DB에 TUNet_realtime 모델에 atafilm 적용: `SITEC_TUNet_realtime_atafilm.py`

### To train

* Run `main.py`:
    ```
    $ python main.py --mode train
    ```
* To train with pretraining model,
  Modify parameters in `CONFIG.task` to ['msm', 'nae', 'nb_bwe' ,'msm+nb_bwe','bwe']
    ```
    $ python main.py --mode train --version {version number of pretrained model}
    ```

## Test

### Audio generation & Evaluation
* Modify parameters in `main.py`
SAVE = True
ONNX = False
ONNX_ZERO = False
CPU = False
SINGLE = False 

* Run `main.py` with a version number to be evaluated:
    ```
    $ python main.py --mode eval --version 5
    ```

  ### ONNX inferencing
  We provide ONNX inferencing scripts and the best ONNX model (converted from the best checkpoint) at `lightning_logs/best_model.onnx`.
  * Convert a checkpoint to an ONNX model:
      ```
      python main.py --mode onnx --version 5
      ```
    The converted ONNX model will be saved to `lightning_logs/version_5/checkpoints`.
  * Put test audios in `test_samples` and inference with the converted ONNX model (see `inference_onnx.py` for more details):
       ```
      python inference_onnx.py
      ```

## Citation
* TUNet paper: https://ieeexplore.ieee.org/abstract/document/9747699
