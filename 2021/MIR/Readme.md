## Mask impulse response (MIR)

마스크로 인한 음성의 명료도 저하 분석을 위하여 마스크 임펄스 응답 Mask impulse response (MIR) 입니다. 본 코드는 2021년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한 "원격다자간 영상회의에서의 음성 품질 고도화 기술개발" 과제의 일환으로 공개된 명료도 향상 부문의 1차년도 MIR 코드입니다. 본 코드는 Matlab과 Python파일로 구분되어 있습니다.

본 코드의 특징은 다음과 같습니다.
* 음성 명료도 분석을 위하여 마스크 임펄스 응답 추정
* 추정된 마스크 임펄스를 convolution하여 마스크 착용/미착용 음성 데이터베이스 구축

영어 음성 TIMIT DB를 사용하여 데이터셋 생성

## Requirements
* Numpy
* h5py
* Matlab

## Prepare exponential sine sweep (ESS) sweeptone
1. Matlab Scipy 모듈을 이용하여 trainset에 사용되어질 LPF DB 생성
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
* TIMIT: https://catalog.ldc.upenn.edu/LDC93s1
