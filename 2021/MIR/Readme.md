## Mask impulse response (MIR)

마스크로 인한 음성의 명료도 저하 분석을 위하여 마스크 임펄스 응답 Mask impulse response (MIR) 입니다. 본 코드는 2021년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한 "원격다자간 영상회의에서의 음성 품질 고도화 기술개발" 과제의 일환으로 공개된 명료도 향상 부문의 1차년도 MIR 코드입니다. 본 코드는 matlab과 python 폴더로 구분되어 있습니다.

본 코드의 특징은 다음과 같습니다.
* 음성 명료도 분석을 위하여 마스크 임펄스 응답 추정
* 추정된 마스크 임펄스를 음성 신호와 convolution하여 마스크 착용/미착용 음성 데이터베이스 구축

영어 음성 TIMIT DB를 사용하여 데이터셋 생성

## Requirements
* Matlab 2022a
* Python 3.8
* Numpy
* Pydub
* Soundfile
* pystoi


## Exponential sine sweep (ESS) sweeptone 생성 및 MIR 추정
1. Matlab 프로그램의 sweeptone() function을 사용하여 ESS 생성
2. 생성된 ESS를 방 환경에서 재생 및 녹음
3. 녹음된 ESS와 원본 ESS를 Matlab 프로그램의 impzest() 함수를 사용하여 마스크 임펄스 응답 추정

Testset은 1번 과정을 생략하고 위의 방식으로 진행하여 dataset 준비.

## Convolution with MIR
to convolve audio samples with multiple MIR
```
python convolve_multi.py
```
to convolve audio smaples with single MIR
```
python convolve_single.py
```
to adjust volume of audio samples
```
python change_volume.py
```

## Evaluate 마스크 명료도 저하 분석
to evaluate synthetic dataset with STOI
```
python stoi.py
```

## Reference
* TIMIT: https://catalog.ldc.upenn.edu/LDC93s1
* C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time Objective Intelligibility Measure for Time-Frequency Weighted Noisy Speech', ICASSP 2010, Texas, Dallas.
