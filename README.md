# WiseKB

### Dependency
- Check the requirements.txt
- torch 버전을 못 찾는 경우 CPU or GPU버전을 https://pytorch.org/get-started/locally/ 에서 받아주시면 됩니다.
- PyTorch == 1.0.1
- torchtext == 0.4.0

```bash
pip install -r requirements.txt
```

## 데이터 전처리  

#### Step 0: setting CONFIG
- config/config.yaml 파일을 수정
- `use` option값을 새로 정의하시거나 기존 정의된값을 쓰시면 1개모듈 or 전체모듈 사용가능 

#### Step 1: 형태소 분석기
- `use : 1` 후 access_key 기입후, 데이터셋 준비후 
```bash
python data_preprocess.py 
```

#### Step 2: Vocab 생성
- `use : 2` 후 형태소분석된 파일 준비후 
```bash
python data_preprocess.py 
```

#### Step 3: 비슷한 발화 찾기
- `use : 3` 후 `model : W` (W: whoosh, ME: maximumentropy) 하여서 사용 
- 본인의 유사발화찾기 모듈로 대체가능 결과물만 `질문 index:유사발화 index` 식으로만 저장하면 됨
```bash
python data_preprocess.py 
```

#### Step 4: Word-to-vec
- `use : 4` 후 vocab파일과 유사발화 찾은 결과물로 생성 
```bash
python data_preprocess.py 
```

#### Step 5: split-data-set
- `use : 5` 후 train-valid ratio설정하고
```bash
python data_preprocess.py 
```

#### Step 6: make test셋
- `use : 6` 후 test셋은 `q-'q`의 비슷함을 측정하여서 `query | | retrieved query | retrieved response`생성
```bash
python data_preprocess.py 
```

## 학습  

#### Train
- skeleton generator: `template` 폴더의 config 설정 후, `./train.sh`
- response generator: `pretrain` 폴더의 config 설정 후, `./train.sh`
- 대화 생성 : `soft`폴더의 config 설정 후, `./train.sh` 결과물 보려면 `translate.sh`

#### skeleton generator 확인
- `template` 폴더의 `.test.sh`을 설정 맞추고 실행하면 0이 마스킹된 결과 

## 형태소->문장 만드는 모델

#### 사용법
- `wise_reporter` 안의 readme, requirement.txt 확인
- 사용법 예제는 `test.py` 확인
- https://github.com/KNU-NLPlab/wise_reporter/ 에 계속 업데이트중

## 유사발화찾기

#### 사용법
- `ME_search` 폴더 안의 `module.py` 를 import하여 사용
- `module.py`의 전역 변수 수정을 통해 데이터 변경 가능
- 동작 흐름
    1. sentence_parse : 입력 문장의 형태소 분석
    2. domain_calssify : 입력 문장의 도메인 분류
    3. extract_similar_sentence : 입력 문장과 비슷한 문장 추출
- 사용법 예제는 `example.py`를 참고