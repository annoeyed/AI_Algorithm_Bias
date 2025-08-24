# 다중 도메인 AI 모델 편향성 심층 분석 프레임워크

이 프로젝트는 컴퓨터 비전, 자연어 처리, 의료 AI 등 **다양한 도메인에 걸쳐** AI 모델에 내재된 편향성을 심층적으로 분석하고, 그 영향을 정량적으로 측정하는 실험 프레임워크를 제공합니다.

현재 **FairFace 얼굴 이미지 데이터셋**을 사용한 전체 분석 파이프라인이 `AIBiasExperiment.ipynb` 주피터 노트북에 구현되어 있습니다. 이 노트북은 다른 데이터셋에도 동일한 분석 방법론(Baseline 평가, 편향 민감도 곡선 등)을 적용할 수 있는 템플릿 역할을 합니다.

##  분석 데이터셋

이 프레임워크는 다음과 같은 세 가지 주요 공정성 벤치마크 데이터셋을 대상으로 합니다.

### 1. FairFace (얼굴 이미지)
- **도메인**: Computer Vision
- **설명**: 인종(7개 그룹), 성별, 연령 분포의 균형을 맞춘 얼굴 속성 데이터셋으로, 안면인식 기술의 공정성 평가에 표준처럼 사용됩니다.
- **거버넌스 포인트**: 안면인식 기술의 인종/성별 간 성능 차별 문제 검증

### 2. FAIRE (이력서 텍스트)
- **도메인**: Natural Language Processing
- **설명**: LLM 기반 이력서 평가 시스템의 성별 및 인종 편향을 측정하기 위한 벤치마크입니다. 채용과 같은 중요한 사회적 의사결정 시스템의 공정성을 검증합니다.
- **거버넌스 포인트**: 채용 과정의 투명성 및 공정성 확보

### 3. Harvard-GF (안과 의료 영상)
- **도메인**: Medical AI
- **설명**: 녹내장 진단을 위한 안과 영상 데이터셋으로, 인종 간 균형을 고려하여 수집되었습니다. 의료 AI가 특정 인종 그룹에 불리하게 작동하지 않는지 실험하는 데 사용됩니다.
- **거버넌스 포인트**: 의료 AI의 환자 그룹별 공정성 및 신뢰성 검증

---

##  시작하기

### 1. 필수 조건 (Prerequisites)

- Python 3.8 이상
- 관련 라이브러리 설치:
  ```bash
  pip install -r requirements.txt
  ```

### 2. 데이터셋 준비

각 데이터셋을 다운로드하여 `data/` 디렉토리 아래에 정리합니다.

```
BLOOM/
└── data/
    ├── fairface/
    ├── FAIRE/
    └── harvard_gf/
```

#### **FairFace 데이터셋 준비 (완료)**

1.  **다운로드**: [공식 GitHub 저장소](https://github.com/joojs/fairface)의 안내에 따라 이미지와 라벨 파일을 다운로드합니다.
2.  **파일 구조**: 다운로드한 파일들을 다음과 같이 배치합니다.
    -   이미지 파일 전체 → `data/fairface/`
    -   `fairface_label_train.csv` → 프로젝트 루트
    -   `fairface_label_val.csv` → 프로젝트 루트

#### **FAIRE 데이터셋 준비**

1.  **저장소 복제**: FAIRE 데이터셋은 공식 GitHub 저장소에 포함되어 있습니다. 다음 명령어로 저장소를 복제합니다.
    ```bash
    git clone https://github.com/athenawen/FAIRE-Fairness-Assessment-In-Resume-Evaluation.git
    ```
2.  **파일 복사**: 복제된 `FAIRE-Fairness-Assessment-In-Resume-Evaluation/data/` 디렉토리의 내용물을 이 프로젝트의 `data/FAIRE/` 디렉토리로 복사합니다.

#### **Harvard-GF 데이터셋 준비**

1.  **접근 요청**: Harvard-GF는 의료 데이터이므로, 사용을 위해 데이터 사용 계약(Data Use Agreement, DUA)을 체결해야 합니다.
2.  **신청 페이지**: [공식 데이터셋 웹사이트](https://ophai.hms.harvard.edu/code/harvard-gf/)에 접속하여 'Request the dataset' 링크를 통해 접근을 요청합니다.
3.  **파일 배치**: 승인 후 다운로드한 데이터셋을 `data/harvard_gf/` 디렉토리에 배치합니다.

### 3. 실험 실행

-   **FairFace 실험**: `AIBiasExperiment.ipynb` 노트북을 열고 순서대로 셀을 실행합니다. 노트북 상단의 경로 설정이 올바른지 확인하세요.
-   **FAIRE 및 Harvard-GF 실험**: 제공된 FairFace 노트북을 템플릿으로 삼아, 각 데이터셋의 특성(데이터 로더, 모델 구조 등)에 맞게 수정하여 새로운 분석 노트북을 생성하고 실행합니다.

##  핵심 분석 방법론

`AIBiasExperiment.ipynb`는 각 데이터셋에 적용할 수 있는 4가지 핵심 분석 방법론을 제시합니다.

1.  **Baseline 모델 구축**: 모델 성능의 기준점을 마련하고 기본적인 편향성을 측정합니다.
2.  **편향 민감도 곡선 (Micro-Bias Sensitivity Curve)**: 데이터셋의 미세한 편향 변화에 모델이 얼마나 민감하게 반응하는지 측정합니다.
3.  **과보정 피해 지수 (Over-Correction Damage Index, ODI)**: 편향 완화 기법 적용 시 발생하는 성능-공정성 간의 Trade-off를 정량적으로 분석합니다.
4.  **잠복 하위집단 탐지 (Hidden Subgroup Discovery)**: 여러 속성이 교차하는 특정 하위집단에서의 잠재적 편향을 탐지합니다.

##  파일 구조

```
BLOOM/
├── AIBiasExperiment.ipynb      # 메인 실험 노트북 (FairFace 기준)
├── data/
│   ├── fairface/               # (데이터셋) FairFace 이미지 파일
│   ├── FAIRE/                  # (데이터셋) FAIRE 텍스트 파일
│   └── harvard_gf/             # (데이터셋) Harvard-GF 의료 영상 파일
├── fairface_label_train.csv    # (데이터셋) FairFace 학습용 라벨
├── fairface_label_val.csv      # (데이터셋) FairFace 검증용 라벨
├── results/                    # 실험 결과(그래프 등)가 저장되는 디렉토리
├── requirements.txt            # 필요한 패키지 목록
└── README.md                   # 이 파일
```

