# Bandwidth extension (BWE)
본 코드는 2024년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한 "원격다자간 영상회의에서의 음성 품질 고도화 기술개발" 과제의 일환으로 공개된 명료도 향상 부문의 4차년도 코드입니다. 

본 코드의 특징은 다음과 같습니다.
* This code is from (https://github.com/Crystalsound/FRN)
* 음성 명료도 저하 극복을 위한 대역폭 확장 BWE 알고리즘 딥러닝 모델 FRN을 baseline으로 선정
* Subband loss 적용을 통한 모델 성능 고도화 

## 1. Setup 
자세한 환경 설치 방법은 위 링크 참고

### Requirements
* PyTorch
* Numpy
* h5py

To install all required packages via pip command
```
Pip3 install -r frn_requirments.txt
```

## 2. Prepare dataset

영어 음성 VCTK DB 및 한국어 음성 SITEC DB 사용
SITEC DB는 다음과 같은 room환경에서 RIR을 통하여 데이터셋 생성

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
* In order to load the datasets, text files that contain training and testing audio paths are required.

## 3. Training

### Configuration for training parameters
config.py 파일 참조

### To train

* Run `main.py`:
    ```
    $ python main.py --mode train
    ```
* To train with pretraining model, run `main.py` with following argument:
    ```
    $ python main.py --mode train --version {version number of pretrained model}
    ```

## 4. Test

### Audio generation & Evaluation

* Run `main.py` with a version number to be evaluated
    ```
    $ python main.py --mode eval --version {version number of trained model}
    ```

## 5. Reference
* FRN paper: https://ieeexplore.ieee.org/document/10097132
