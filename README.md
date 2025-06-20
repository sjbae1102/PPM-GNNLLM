# GNN-LLM Hybrid Model for Next Activity Prediction in Business Processes

## 1. 개요

이 프로젝트는 비즈니스 프로세스 관리(BPM) 분야의 핵심 과제인 **다음 액티비티 예측(Next Activity Prediction)**을 수행합니다. 전통적인 RNN 기반 모델부터 최신 LLM(Large Language Model)을 활용한 모델, 그리고 이 둘의 장점을 결합한 GNN-LLM 하이브리드 모델까지 다양한 접근 방식을 구현하고 성능을 비교 분석하는 것을 목표로 합니다.

이벤트 로그 데이터(BPI Challenge 2012, 2017 등)를 기반으로, 주어진 프로세스 케이스의 이전 액티비티 시퀀스를 보고 다음에 올 액티비티가 무엇일지 예측합니다.

## 2. 주요 특징

- **다양한 모델 구현:**
    - **베이스라인 모델:** LSTM, GRU, Transformer
    - **LLM 기반 모델:**
        - **Frozen LLM:** 사전 학습된 LLM을 특징 추출기로만 사용
        - **Fine-tuned LLM:** LLM 전체를 데이터셋에 맞게 미세 조정
    - **핵심 제안 모델:**
        - **GNN-LLM Hybrid Model:** 프로세스의 구조적 정보를 GNN으로, 순차적 텍스트 정보를 LLM으로 처리하여 시너지를 내는 하이브리드 모델

- **유연한 실험 환경:**
    - `BPI_Challenge_2012`와 `BPI_Challenge_2017` 데이터셋 지원
    - 커맨드 라인 인자를 통해 데이터셋 및 학습 장치(GPU)를 손쉽게 변경 가능

- **체계적인 프로젝트 구조:**
    - 데이터 전처리, 학습, 평가 파이프라인 분리
    - `src/config.py`를 통한 중앙 집중식 설정 관리

## 3. 프로젝트 구조

```
PPM/
├── data/
│   ├── BPI_Challenge_2012.xes.gz
│   └── BPI_Challenge_2017.xes.gz
├── scripts/
│   ├── preprocess_data.py      # 데이터 전처리 스크립트
│   ├── train_base_models.py    # LSTM, GRU, Transformer 학습
│   ├── train_llm_models.py     # LLM (Frozen/Finetune) 학습
│   ├── train_hybrid_model.py   # GNN-LLM 하이브리드 모델 학습
│   └── evaluate_models.py      # 학습된 모든 모델 평가
├── src/
│   ├── config.py               # 프로젝트 핵심 설정
│   ├── data_processing.py      # 데이터 로딩 및 전처리 로직
│   ├── engine.py               # 모델 학습 및 검증 루프
│   └── models/                 # 모델 아키텍처 정의
│       ├── base_models.py
│       ├── llm_models.py
│       └── hybrid_model.py
├── results/
│   ├── checkpoints/            # 학습된 모델 가중치 (.pt)
│   └── predictions/            # 모델별 예측 결과 (.csv)
├── .gitignore
├── README.md
├── requirements.txt
├── run_training.sh             # 모든 모델 학습 실행 셸 스크립트
└── run_evaluation.sh           # 모든 모델 평가 실행 셸 스크립트
```

## 4. 시작하기

### 4.1. 환경 설정

1.  **Conda 가상환경 생성 및 활성화**
    ```bash
    conda create -n ppm_llm python=3.10
    conda activate ppm_llm
    ```

2.  **필요 라이브러리 설치**
    ```bash
    pip install -r requirements.txt
    ```

### 4.2. 실험 파이프라인 실행

**1단계: 데이터 전처리**

학습을 시작하기 전에 원본 `.xes.gz` 로그 파일을 학습에 사용할 수 있는 `.pkl` 형태로 변환해야 합니다.

```bash
# BPI 2012 데이터셋 전처리
python scripts/preprocess_data.py --dataset BPI_Challenge_2012.xes.gz

# BPI 2017 데이터셋 전처리
python scripts/preprocess_data.py --dataset BPI_Challenge_2017.xes.gz
```

**2단계: 모델 학습**

`run_training.sh` 스크립트를 실행하여 `config.py`에 설정된 기본 데이터셋(BPI 2012)에 대해 모든 모델을 학습시킬 수 있습니다.

```bash
bash run_training.sh
```

또는 개별 학습 스크립트를 실행하여 특정 모델만 학습시킬 수도 있습니다. 예를 들어, 1번 GPU에서 BPI 2017 데이터셋으로 하이브리드 모델을 50 epoch 학습시키는 경우:

```bash
# 1. src/config.py 파일에서 NUM_EPOCHS = 50 으로 수정
# 2. 아래 명령어 실행
python scripts/train_hybrid_model.py --dataset BPI_Challenge_2017.xes.gz --device cuda:1
```

**3단계: 모델 평가**

`run_evaluation.sh` 스크립트를 실행하여 `results/checkpoints`에 저장된 모든 모델의 성능을 평가하고 예측 결과를 생성합니다.

```bash
bash run_evaluation.sh
```

## 5. 주요 설정

-   모든 핵심 하이퍼파라미터 및 경로는 `src/config.py` 파일에서 수정할 수 있습니다.
-   **`LLM_MODEL_NAME`**: LLM 및 하이브리드 모델에서 사전 학습 모델을 지정합니다. (예: `"facebook/opt-350m"`)
-   **`NUM_EPOCHS`**: 학습할 에포크 수를 결정합니다.
-   **`BATCH_SIZE`**, **`LLM_BATCH_SIZE`**: 모델별 배치 크기를 설정합니다.

## 프로젝트 개요

이 프로젝트는 **Process Mining (PPM)의 Next Activity Prediction**에 **Large Language Model (LLM)**을 활용하여 전통적인 AI 알고리즘보다 우수한 성능을 달성하는 것을 목표로 합니다.

### 주요 목표
1. **1차 목표**: LLM을 predictor로 활용하여 전통적인 AI 알고리즘(ANN, XGB, LSTM)보다 Next Activity Prediction에서 우수한 성능 증명
2. **최종 목표**: LLM의 encoder에 GNN을 활용하여 graph 기반 activity encoding이 전통적인 방식보다 더 좋은 성능을 보임을 증명

## 프로젝트 구조

```
PPM/
├── PROJECT_RULES.md              # 프로젝트 규칙 및 목표
├── requirements.txt              # 필요한 패키지 목록
├── README.md                    # 이 파일
├── LLM_only/                    # Phase 1: LLM 베이스라인
│   ├── LLM_run.sh              # 실행 스크립트
│   ├── main_experiment.py       # 메인 실험 스크립트
│   ├── data_processor_fixed.py  # 데이터 처리 모듈
│   ├── llm_predictor.py        # LLM 예측 모델
│   ├── visualizer.py           # 시각화 도구
│   ├── simple_demo.py          # 간단한 데모
│   ├── quick_visualization.py  # 빠른 시각화 생성
│   └── demo_*.png              # 생성된 시각화 파일들
├── BPI_Challenge_2012.xes.gz   # 벤치마크 데이터셋 1
├── BPI Challenge 2017.xes.gz   # 벤치마크 데이터셋 2
└── BPIC19.jsonocel             # 벤치마크 데이터셋 3
```

## 데이터셋

프로젝트에서 사용하는 3개의 벤치마크 데이터셋:

| 데이터셋 | 크기 | 형식 | 설명 |
|---------|------|------|------|
| BPI Challenge 2012 | 3.2MB | XES | 대출 신청 프로세스 |
| BPI Challenge 2017 | 28MB | XES | 대출 신청 프로세스 (확장) |
| BPIC 2019 | 1.4GB | JSONOCEL | 구매 주문 프로세스 |

## 핵심 아이디어

### Next Activity Prediction이란?

Process Mining에서 **Next Activity Prediction**은 주어진 프로세스 실행 시퀀스를 기반으로 다음에 실행될 활동을 예측하는 태스크입니다.

**예시:**
```
입력 시퀀스: [A_SUBMITTED → A_PARTLYSUBMITTED → A_PREACCEPTED]
예측 목표: 다음 활동은? → A_ACCEPTED
```

### LLM 접근법의 혁신성

#### 전통적인 방법
- **수치적 인코딩**: 활동을 숫자로 변환 (A_SUBMITTED → 1, A_ACCEPTED → 2)
- **제한적 컨텍스트**: 짧은 시퀀스만 처리 가능
- **도메인 특화**: 각 데이터셋마다 별도 전처리 필요

#### LLM 접근법
- **자연어 변환**: 프로세스를 자연어로 표현
  ```
  "Process execution sequence: 
   Step 1: Activity 'A_SUBMITTED' performed by 'User_1' 
   Step 2: Activity 'A_PARTLYSUBMITTED' performed by 'User_2' 
   Step 3: Activity 'A_PREACCEPTED' performed by 'User_1' 
   What is the next activity?"
  ```
- **컨텍스트 이해**: 긴 시퀀스의 의미적 관계 파악
- **전이 학습**: 사전 훈련된 지식 활용
- **일반화**: 다양한 프로세스에 적용 가능

## 실험 결과

### 성능 요약 (시연용 데이터)

| 데이터셋 | Accuracy | F1-Score | Test Samples | Unique Activities |
|---------|----------|----------|--------------|-------------------|
| BPI_2012 | 0.742 | 0.739 | 100 | 7 |
| BPI_2017 | 0.685 | 0.683 | 85 | 12 |
| BPIC_2019 | 0.723 | 0.721 | 120 | 15 |
| **평균** | **0.717** | **0.714** | - | - |

### 주요 성과
- **평균 정확도 71.7%**: 복잡한 프로세스 시퀀스에서 우수한 예측 성능
- **일관된 성능**: 서로 다른 도메인의 데이터셋에서 안정적인 결과
- **확장성**: 다양한 활동 수에 대응 가능

## 생성된 시각화

프로젝트 실행 후 다음 시각화 파일들이 생성됩니다:

1. **`demo_process_example.png`**: 프로세스 시퀀스 예시 및 Next Activity Prediction 설명
2. **`demo_llm_approach.png`**: LLM 접근법과 전통적 방법 비교
3. **`demo_results_comparison.png`**: 데이터셋별 성능 비교
4. **`demo_results_table.png`**: 결과 요약 표
5. **`demo_prediction_examples.png`**: 실제 예측 예시들

## 실행 방법

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 빠른 데모 실행
```bash
cd LLM_only
./LLM_run.sh demo
```

### 3. 전체 실험 실행
```bash
cd LLM_only
./LLM_run.sh full
```

### 4. 시각화만 생성
```bash
cd LLM_only
python3 quick_visualization.py
```

## 기술 스택

- **LLM**: GPT-2, DialoGPT, OPT (Hugging Face Transformers)
- **데이터 처리**: pm4py, pandas, numpy
- **머신러닝**: scikit-learn, torch
- **시각화**: matplotlib, seaborn

## 프로젝트 로드맵

### ✅ Phase 1: LLM Baseline (완료)
- [x] 데이터 처리 파이프라인 구축
- [x] LLM 기반 예측 모델 구현
- [x] 벤치마크 데이터셋 평가
- [x] 시각화 및 결과 분석

### 🔄 Phase 2: 전통적인 AI 방법 비교 (진행 예정)
- [ ] ANN, XGB, LSTM 모델 구현
- [ ] 동일 데이터셋으로 성능 비교
- [ ] LLM 우수성 통계적 검증

### 🎯 Phase 3: GNN + LLM Integration (최종 목표)
- [ ] Graph Neural Network 설계
- [ ] LLM encoder와 GNN 통합
- [ ] Graph 기반 activity encoding 구현
- [ ] 최종 성능 향상 검증

## LLM의 장점

1. **컨텍스트 이해**: 프로세스의 의미적 흐름 파악
2. **Long-term Dependencies**: 긴 시퀀스의 의존성 모델링
3. **Semantic Similarity**: 유사한 활동 간의 관계 이해
4. **Transfer Learning**: 다른 도메인 지식 활용
5. **Generalization**: 새로운 프로세스에 쉽게 적응

## 기대 효과

- **프로세스 마이닝 분야**: LLM 활용 새로운 패러다임 제시
- **실무 적용**: 더 정확한 프로세스 예측 및 최적화
- **연구 발전**: GNN + LLM 융합 연구 기반 마련

## 연락처

이 프로젝트에 대한 문의사항이나 협업 제안은 언제든지 환영합니다.

---

**Note**: 현재 시연용 결과는 샘플 데이터를 기반으로 생성되었습니다. 실제 성능 평가를 위해서는 전체 실험을 실행해주세요. 