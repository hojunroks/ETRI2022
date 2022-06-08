# 사용자 일상 학습 모델의 예측 불확실성에 기반한 비일상 이벤트 탐지 기술


## Setup
    # Install python dependencies
    pip install -r requirements.txt
    
## 데이터셋
아래 링크의 데이터 파일 다운로드

ETRI 라이프로그 데이터셋
https://nanum.etri.re.kr/share/schung1/ETRILifelogDataset2020?lang=ko_KR

사용된 데이터 파일 목록
- user01-06 data
- user07-10 data
- user11-12 data
- user21-25 data
- user26-30 data

## 데이터 전처리 내용

 - 인접한 시점에서 상태의 변화가 없는 경우는 하나의 상태로 통합
 - 행동과 장소 데이터는 범주형 데이터, 감정은 척도에 따른 수치형 데이터로 구분
 - 범주형 데이터의 경우 유저별로 응답한 총 응답 가지수에 따라 이를 one-hot vector로 표현하였고, 수치형 데이터는 값 그대로 이용
 - 각 사용자의 데이터 중 초기 67%의 데이터는 학습 데이터로, 후기 33% 데이터는 테스트 데이터로 분리
 - 데이터 경로는 /data/etri_lifelog 로 가정

## Train

#### 학습 실행 예시 및 중요 arguments 

데이터 경로가 /data/etri_lifelog 일 경우,

    ## MLP & user 1
    python train.py --model_name MLP --data_dir /data/etri_lifelog --person_index 1 
    
    ## LSTM & user 1
    python train.py --model_name LSTM --data_dir /data/etri_lifelog --person_index 1
    
    ## Transformer & user 1
    python train_transformer.py --data_dir /data/etri_lifelog --person_index 1
    



### Tensorboard 실행

실험 결과는 runs 디렉토리에 저장

    tensorboard --logdir=runs --bind_all


