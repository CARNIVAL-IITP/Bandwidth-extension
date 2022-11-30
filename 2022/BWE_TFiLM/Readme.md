# Bandwidth extension (BWE)
본 코드는 2021년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한 "원격다자간 영상회의에서의 음성 품질 고도화 기술개발" 과제의 일환으로 공개된 명료도 향상 부문의 2차년도 코드입니다. 

본 코드의 목적은 다음과 같습니다.
*마스크의 명료도 저하 극복 알고리즘 개발
*명료도 지표를 직접 목적함수에 적용하는 음성 명료도 향상 알고리즘 개발

## Requirements
* Tensorflow
* Numpy
* h5py
To install all required packages via pip command
```
Pip3 install -r tfilm_tf_requirements.txt
```

## Prepare dataset
1. 2021/MIR 폴더를 참고하여 마스크 착용/미착용 데어터셋 생성
2. h5_data.py를 사용하여 .h5파일형식의 trainset과 test set 각각 생성

Testset은 1번 과정을 생략하고 위의 방식으로 진행하여 dataset 준비.

## Training
to train the model
```
python train.py
```
to train with STOI loss
```
python train_stoi.py
```
to train without memory leak
```
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 python3 train.py
```
* https://koodev.tistory.com/77

## Test
To inference
```
python test.py
```
To evaluate 
```
python evaluate.py
```

## Reference
* Baseline: https://github.com/leolya/Audio-Super-Resolution-Tensorflow2.0-TFiLM
* TIMIT: https://catalog.ldc.upenn.edu/LDC93s1
* VCTK: https://datashare.ed.ac.uk/handle/10283/2950
* 메모리 누수: https://github.com/tensorflow/tensorflow/issues/44176#issuecomment-783768033
