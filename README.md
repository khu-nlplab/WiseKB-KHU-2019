# WiseKB

### Dependency
- Check the requirements.txt
- torch 버전을 못 찾는 경우 CPU or GPU버전을 https://pytorch.org/get-started/locally/ 에서 받아주시면 됩니다.
- PyTorch == 1.0.1
- torchtext == 0.4.0

```bash
pip install -r requirements.txt
```

### 데이터 전처리  

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
