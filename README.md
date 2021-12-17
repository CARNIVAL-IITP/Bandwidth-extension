# BWE
Baseline을 개선하여 음성의 명료도 저하를 극복하기 위한 딥러닝 기반 고주파 대역폭 확장 Bandwidth Extension (BWE) 입니다. 본 코드는 2021년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한 "원격다자간 영상회의에서의 음성 품질 고도화 기술개발" 과제의 일환으로 공개된 명료도 향상 부문의 1차년도 코드입니다. 본 코드는 Matlab과 Python파일로 구분되어 있습니다.

본 모델의 특징은 다음과 같습니다.
* 음성 명료도 향상을 위하여 frequency domain에서 BWE를 행하는 Sequnetial neural network 기반의 기술
* 새로운 목적함수 Huber loss와 Dropout적용을 통한 성능향상

TIMIT DB로 훈련하였으며 train set은 low pass filter를 통과하여 고역대(4 kHz ~ 8 kHz)가 소실된 4000개의 문장을 사용하였습니다. Test set은 저역대와 고역대(0 kHz ~ 8 kHz)가 모두 온전한 1800개의 문장을 사요하였습니다. Low pas filter는 FIR 필터 type I, 필터계수 101, cutoff 0.5를 파이썬 Scipy 모듈을 사용하여 구현하였습니다.

## Requirements
* Tensorflow
* Numpy
* h5py
* Matlab

## Prepare dataset
1. 파이썬 Scipy 모듈을 이용하여 trainset에 사용되어질 LPF DB 생성
2. 생성된 LPF된 DB들을 BWE/Matlab/Features_extraction_FFT64.m 이용하여 feature정보가 있는 .mat형식의 파일 생성
3. Normalize_multiple_files.m 이용하여 feature들을 normalize 및 shuffle 진행.

Testset은 1번 과정을 생략하고 위의 방식으로 진행하여 dataset 준비.

## Training
to train the regression model with multiple .mat dataset, run the following command
```
python tf1.0_DNN_regression_multiple.py
```
to train the classification model with single .mat dataset, run the following command
```
python tf1.0_DNN_classification.py
```
to train the regression model with single .mat dataset, run the following command
```
python tf1.0_DNN_regression_single.py
```

## Test
to test the model, run BWE/Matlab/Synthesis_FFT64.m

## Reference
* Baseline: https://www.sciencedirect.com/science/article/pii/S1051200420301056
* TIMIT: https://catalog.ldc.upenn.edu/LDC93s1
