# ETTh1 Direct Multi-Model Forecasting 실험 및 개선 이력

## 1. 문서 목적

이 문서는 `ett_multimodel_kaggle.py` 기반 ETTh1 직접 예측 파이프라인의 설계, 실행 결과, 판단 근거, 코드 변경 사항을 시간 순서대로 기록한다.

기존 `MODEL_EXPERIMENT_HISTORY.md`는 seasonal anchor와 residual 보정 실험을 기록한다. 이 문서는 다음 구조를 별도로 추적한다.

```text
multivariate past window
  -> Seasonal Naive anchor
  -> PatchTST residual forecast
  -> future OT T0~T95
```

핵심 목적은 단순히 모델 목록을 나열하는 것이 아니다. 각 단계에서 다음 내용을 추적할 수 있도록 작성했다.

1. 어떤 누수 방지 규칙을 적용했는가.
2. 어떤 직접 예측 모델을 구현했는가.
3. 모델별 validation 결과가 어땠는가.
4. 앙상블이 단일 모델보다 좋아졌는가.
5. 다음 실험에서 무엇을 유지하고 무엇을 수정할 것인가.

새 대화 세션에서 직접 예측 멀티모델 작업을 이어갈 때는 이 문서를 먼저 읽는다.

---

## 2. 문제 정의와 점수 해석

### 2.1 데이터셋

사용 데이터는 `ETTh1.csv`다.

- timestamp 수: `17,420`
- 주기: hourly
- raw 변수 수: `7`
- raw 변수:
  - `HUFL`
  - `HULL`
  - `MUFL`
  - `MULL`
  - `LUFL`
  - `LULL`
  - `OT`
- target: `OT`
- 예측 horizon: `96`시간
- 제출 row 수: `142`
- 제출 시작 시점: `2018-02-01 00:00:00`
- 제출 종료 시점: `2018-06-22 00:00:00`

각 제출 row는 특정 날짜 자정부터 시작하는 `T0~T95` 예측값을 포함한다.

### 2.2 직접 예측 구조

기본 입력은 multivariate input이다.

```python
USE_MULTIVARIATE_INPUT = True
FEATURE_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TARGET_COL = "OT"
HORIZON = 96
```

초기 absolute direct multi-model 실험은 raw 변수에 다음 시간 파생 변수를 추가했다.

```text
hour
dayofweek
month
dayofyear
sin_hour
cos_hour
sin_dayofweek
cos_dayofweek
sin_dayofyear
cos_dayofyear
```

총 model feature 수:

```text
17
```

PatchTST residual 기본 경로는 test 기간 외삽에 취약한 raw ordinal calendar를 제외한다.

```text
제외:
  hour
  dayofweek
  month
  dayofyear

유지:
  sin_hour
  cos_hour
  sin_dayofweek
  cos_dayofweek
  sin_dayofyear
  cos_dayofyear
```

PatchTST residual 기본 model feature 수:

```text
13
```

과거 `17`개 feature 비교가 필요하면 `--include-raw-calendar` 옵션을 사용한다.

`--univariate-input` 옵션을 사용하면 입력을 `OT` 하나로 제한할 수 있다.

### 2.3 raw MSE 해석

validation MSE는 정규화된 benchmark MSE가 아니라 `OT` 원 단위의 raw MSE다.

따라서 PatchTST 논문이나 ETTh1 benchmark 표의 normalized MSE와 직접 비교하지 않는다. 이 파이프라인 내부에서 동일 split과 동일 raw scale을 사용한 후보끼리 비교한다.

---

## 3. 파일 구조

### 3.1 메인 스크립트

파일:

```text
ett_multimodel_kaggle.py
```

주요 역할:

- ETTh1 CSV 로딩
- hourly reindex와 causal forward fill
- 시간 파생 변수 생성
- leakage-safe train / validation / test origin 생성
- train 구간 scaler fit
- PyTorch 직접 예측 모델 학습
- 선택적 LightGBM lag-feature 모델 학습
- fixed-weight ensemble
- validation inverse-MSE weighted ensemble
- validation 결과 출력
- 제출 CSV 생성
- 실행 JSON report 생성

### 3.2 산출물 디렉터리

직접 예측 멀티모델 산출물은 다음 경로에 저장한다.

```text
multimodel_output/
```

기본 산출물:

```text
multimodel_output/submit.csv
multimodel_output/run_report.json
```

`multimodel_output/*`는 재생성 가능한 산출물이므로 Git에서 제외한다.

### 3.3 기존 residual 파이프라인과의 분리

기존 seasonal anchor + residual 실험은 다음 문서에서 관리한다.

```text
mdFiles/MODEL_EXPERIMENT_HISTORY.md
```

두 파이프라인은 직접 비교할 수 있지만 구현 흐름을 섞어서 기록하지 않는다.

---

## 4. 누수 방지 설계

### 4.1 supervised target 제한

학습과 validation의 실제 `OT` target은 다음 시점을 넘지 않는다.

```text
2018-01-31 23:00:00
```

예측 horizon이 `96`시간이므로 허용 가능한 마지막 supervised origin은 다음과 같다.

```text
2018-01-28 00:00:00
```

### 4.2 train / validation split

supervised origin은 자정 시점만 사용한다.

```text
allowed supervised origins = 555
train samples = 441
validation samples = 111
```

train target 범위:

```text
y_start = 2016-07-23 00:00:00 ~ 2017-10-06 00:00:00
y_end   = 2016-07-26 23:00:00 ~ 2017-10-09 23:00:00
```

validation target 범위:

```text
y_start = 2017-10-10 00:00:00 ~ 2018-01-28 00:00:00
y_end   = 2017-10-13 23:00:00 ~ 2018-01-31 23:00:00
```

train과 validation target이 겹치지 않도록 purge를 적용한다.

```text
last_train_origin + HORIZON <= first_validation_origin
```

### 4.3 scaler 제한

scaler는 validation 시작 이전 train row에만 fit한다.

```text
scaler fit end exclusive = 2017-10-10 00:00:00
target mean = 16.986237
target std = 8.345091
```

validation과 test에는 transform만 적용한다.

### 4.4 test inference 분리

test inference는 supervised dataset과 별도 경로를 사용한다.

```text
InferenceWindowDataset
```

이 dataset은 입력 `X`와 origin만 받는다. 미래 `y` 배열을 받거나 참조하지 않는다.

각 제출 timestamp에 대해 다음 조건을 assert로 확인한다.

```text
X 마지막 시점 = 제출 timestamp - 1 hour
```

예시:

```text
2018-02-01 00:00:00 예측 X 마지막 시점 = 2018-01-31 23:00:00
2018-06-22 00:00:00 예측 X 마지막 시점 = 2018-06-21 23:00:00
```

---

## 5. 구현 모델

### 5.1 PatchTST

메인 모델이다.

두 context length를 학습하고 평균낸다.

```text
PatchTST prediction =
  0.6 * PatchTST-512 prediction
  + 0.4 * PatchTST-336 prediction
```

공통 설정:

```text
patch length = 16
stride = 8
d_model = 64
n_heads = 4
encoder layers = 2
dropout = 0.2
learning rate = 3e-4
```

seed ensemble 옵션:

```text
--patch-seeds 42,2024,2026
```

### 5.2 DLinear

trend와 seasonal component를 분리한 뒤 선형 projection으로 미래 `96`시간을 예측한다.

```text
input size = 336
learning rate = 1e-3
```

PatchTST와 다른 단순한 inductive bias를 제공하기 위해 포함했다.

### 5.3 N-BEATS

MLP block이 backcast residual을 반복 제거하고 forecast를 누적한다.

```text
input size = 336
hidden size = 256
blocks = 3
dropout = 0.1
learning rate = 1e-3
```

### 5.4 N-HiTS

여러 pooling scale을 사용해 서로 다른 시간 해상도의 forecast를 누적한다.

```text
input size = 336
pool size = 24, 4, 1
dropout = 0.1
learning rate = 1e-3
```

### 5.5 Seasonal Naive

최근 `24`시간 `OT`를 미래 `96`시간에 반복한다.

```text
season length = 24
prediction = tile(last_24_hours, 4)
```

복잡한 모델이 흔들릴 때 앙상블 분산을 낮추는 안정적인 기준점으로 포함했다.

### 5.6 LightGBM lag-feature 모델

`lightgbm`이 설치되어 있으면 선택적으로 학습한다.

사용 feature:

- raw feature lag:
  - `1, 2, 3, 6, 12, 24, 48, 72, 96, 168, 336`
- rolling stats:
  - `6, 12, 24, 48, 96, 168, 336`
- origin calendar feature
- target horizon calendar feature
- horizon index와 sin/cos

이 모델도 origin 이전 관측값만 feature로 사용한다.

---

## 6. 앙상블 설계

### 6.1 fixed-weight ensemble

기본 fixed weight:

```text
PatchTST        = 0.40
DLinear         = 0.25
N-BEATS         = 0.15
N-HiTS          = 0.10
Seasonal Naive  = 0.10
```

LightGBM 포함 fixed weight:

```text
PatchTST        = 0.35
DLinear         = 0.20
N-BEATS         = 0.10
N-HiTS          = 0.10
Seasonal Naive  = 0.10
LightGBM        = 0.15
```

### 6.2 inverse-MSE weighted ensemble

validation MSE가 낮은 모델에 더 높은 weight를 부여한다.

```python
weights = 1 / (mse_values + 1e-8)
weights = weights / weights.sum()
```

특정 모델에 weight가 과도하게 몰리지 않도록 최종 normalized weight에 `0.5` cap을 적용한다.

```text
max weight = 0.5
```

---

## 7. validate-only smoke test

학습 없이 누수 규칙과 제출 스키마를 확인하기 위해 다음 명령을 실행했다.

```bash
python ett_multimodel_kaggle.py --validate-only
```

확인 결과:

```text
last allowed supervised origin = 2018-01-28 00:00:00
last allowed supervised y_end = 2018-01-31 23:00:00
test rows = 142
prediction shape = (142, 96)
submit columns = ID, T0, ..., T95
```

Seasonal Naive validation:

```text
MSE = 6.874173
MAE = 2.033128
```

산출물:

```text
multimodel_output/submit.csv
multimodel_output/run_report.json
```

---

## 8. PatchTST 3-seed full 실행

### 8.1 실행 명령

Linux `.venv`에서 다음 명령을 실행했다.

```bash
python ett_multimodel_kaggle.py --patch-seeds 42,2024,2026
```

실행 설정:

```text
input mode = multivariate + time features
epochs = 30
patience = 5
batch size = 64
validation fraction = 0.20
PatchTST seeds = 42, 2024, 2026
LightGBM optional = enabled
```

생성된 report:

```text
multimodel_output/run_report.json
```

report 생성 시각:

```text
2026-06-02T05:36:13.121013+00:00
```

### 8.2 PatchTST seed별 결과

`PatchTST-512`:

| seed | best epoch | best validation MSE | early stopping | 실행 epoch 수 |
| ---: | ---: | ---: | --- | ---: |
| `42` | `16` | `7.936456` | Yes | `21` |
| `2024` | `10` | `8.060454` | Yes | `15` |
| `2026` | `12` | `7.851808` | Yes | `17` |

`PatchTST-336`:

| seed | best epoch | best validation MSE | early stopping | 실행 epoch 수 |
| ---: | ---: | ---: | --- | ---: |
| `42` | `13` | `7.532903` | Yes | `18` |
| `2024` | `14` | `7.614750` | Yes | `19` |
| `2026` | `5` | `7.873115` | Yes | `10` |

관찰:

1. `336`시간 context가 `512`시간 context보다 seed별 validation에서 대체로 좋았다.
2. seed에 따라 best epoch와 MSE 변동이 있다.
3. seed 평균과 두 lookback blend를 유지할 근거가 있다.

### 8.3 모델별 validation 결과

| 후보 | validation MSE | MAE |
| --- | ---: | ---: |
| PatchTST | `6.588251` | `2.039613` |
| DLinear | `12.760802` | `2.670443` |
| N-BEATS | `14.818713` | `3.096292` |
| N-HiTS | `10.762184` | `2.555611` |
| Seasonal Naive | `6.874173` | `2.033128` |
| LightGBM optional | `28.209537` | `4.393026` |
| Fixed Ensemble | `7.126373` | `2.031226` |
| Inverse-MSE Weighted Ensemble | `6.212227` | `1.890344` |
| Fixed Ensemble with GBM | `8.224663` | `2.192813` |
| Inverse-MSE Weighted Ensemble with GBM | `6.722713` | `1.968025` |

best single model:

```text
PatchTST
validation MSE = 6.588251
```

best overall predictor:

```text
Inverse-MSE Weighted Ensemble
validation MSE = 6.212227
validation MAE = 1.890344
```

### 8.4 최종 제출 weight

LightGBM을 제외한 inverse-MSE weighted ensemble이 선택됐다.

```text
PatchTST        = 0.283170
DLinear         = 0.146197
N-BEATS         = 0.125894
N-HiTS          = 0.173347
Seasonal Naive  = 0.271392
```

submission:

```text
path = multimodel_output/submit.csv
rows = 142
columns = 97
sha256 = 3265549a429dd6ccf51d9a109b540ec545ebd7a5276e21f13cce6c23bd7f5a63
```

### 8.5 판단

첫 full 실행에서 확인한 핵심:

1. PatchTST는 직접 예측 단일 모델 중 가장 좋았다.
2. Seasonal Naive는 PatchTST보다 MSE는 약간 나쁘지만 MAE는 비슷했다.
3. fixed weight는 성능이 좋지 않았다.
4. validation 기반 inverse-MSE weight는 PatchTST 단독과 Seasonal Naive보다 좋아졌다.
5. LightGBM direct lag-feature 모델은 현재 feature와 hyperparameter에서 실패했다.
6. LightGBM을 앙상블에 포함하면 오히려 성능이 나빠졌다.
7. DLinear, N-BEATS, N-HiTS는 단일 모델 성능은 약하지만 capped inverse-MSE ensemble에서 보조 역할을 했다.

주의:

- 이 결과에는 rolling backtest가 아직 없다.
- Kaggle public MSE도 아직 연결하지 않았다.
- validation 개선만 보고 residual 파이프라인보다 우수하다고 결론 내리지 않는다.

---

## 9. residual 파이프라인과 현재 비교

직접 예측 멀티모델과 기존 residual 파이프라인은 validation split과 feature 구조가 다르므로 숫자를 완전히 동일한 조건의 대결로 해석하면 안 된다.

현재 기록:

| 파이프라인 | 후보 | validation MSE | 비고 |
| --- | --- | ---: | --- |
| direct multi-model | inverse-MSE weighted ensemble | `6.212227` | 별도 direct split |
| direct multi-model | PatchTST | `6.588251` | best single |
| seasonal residual | LightGBM residual | `5.713983` | 최신 tree main split |
| seasonal residual | 과거 CatBoost best | `5.658024` | 과거 기록 best |

현재 판단:

- Kaggle 제출 1순위는 검증 이력이 더 풍부한 seasonal residual 후보다.
- direct multi-model은 독립적인 보완 후보로 유지한다.
- direct multi-model 제출 파일은 public score를 연결해 실제 일반화 성능을 확인할 가치가 있다.

---

## 10. 현재 개선 우선순위

### 10.1 rolling backtest 추가

direct multi-model에도 시점 이동 안정성 진단이 필요하다.

우선순위:

1. Seasonal Naive
2. PatchTST
3. inverse-MSE weighted ensemble

각 rolling fold에서 scaler를 해당 train 구간에만 다시 fit해야 한다.

### 10.2 direct LightGBM 재검토

현재 LightGBM direct MSE:

```text
28.209537
```

앙상블에 도움이 되지 않았다.

다음 선택지:

1. 우선 `--skip-gbm`으로 제외한다.
2. direct target 대신 Seasonal Naive residual을 예측하도록 바꾼다.
3. horizon별 모델 또는 horizon block별 모델을 검토한다.

현재는 1번이 기본이다.

### 10.3 PatchTST context 비교

seed별로 `336`시간 context가 대체로 좋았다.

다음 실험:

```text
PatchTST blend:
  current = 0.6 * 512 + 0.4 * 336
  compare = 0.4 * 512 + 0.6 * 336
  compare = 1.0 * 336
```

weight는 validation 기준으로 비교하되 rolling backtest가 추가된 뒤 결정한다.

### 10.4 약한 직접 예측 모델 검토

DLinear, N-BEATS, N-HiTS는 단일 MSE가 높다.

현재 inverse-MSE cap 때문에 약한 모델에도 일정 weight가 남는다. 다음 실험에서는 다음을 비교한다.

```text
all neural + Seasonal Naive
PatchTST + Seasonal Naive
PatchTST + Seasonal Naive + N-HiTS
```

단순한 앙상블이 더 나으면 약한 모델을 제거한다.

---

## 11. 로컬 실행 방법

가상환경 활성화:

```bash
cd /home/sangw/2026_DeepSyS
source .venv/bin/activate
```

누수 규칙과 제출 스키마만 점검:

```bash
python ett_multimodel_kaggle.py --validate-only
```

빠른 학습 smoke 실행:

```bash
python ett_multimodel_kaggle.py \
  --quick \
  --output multimodel_output/submit_quick.csv \
  --report-output multimodel_output/run_report_quick.json
```

PatchTST 3-seed full 실행:

```bash
python ett_multimodel_kaggle.py \
  --patch-seeds 42,2024,2026
```

GPU 메모리가 부족하면:

```bash
python ett_multimodel_kaggle.py \
  --patch-seeds 42,2024,2026 \
  --batch-size 32
```

---

## 12. 새 세션에서 이어서 작업할 때 확인할 것

1. `MULTIMODEL_EXPERIMENT_HISTORY.md`를 읽는다.
2. `multimodel_output/run_report.json`을 확인한다.
3. `multimodel_output/submit.csv` checksum을 확인한다.
4. Kaggle에 제출했다면 public MSE를 기록한다.
5. direct multi-model rolling backtest 구현 여부를 확인한다.
6. `--skip-gbm` 기준 실험을 우선 검토한다.
7. PatchTST `336`시간 context 비중 확대 실험을 검토한다.

현재 기록상 주요 결과:

```text
best direct single model = PatchTST, validation MSE 6.588251
best direct overall = inverse-MSE weighted ensemble, validation MSE 6.212227
direct Seasonal Naive = validation MSE 6.874173
direct LightGBM = validation MSE 28.209537, 현재 제외 권장
legacy direct ensemble Kaggle public MSE = 사용자 보고 기준 13점대
```

---

## 13. PatchTST residual 기본 경로 추가

이전 직접 멀티모델 비교는 validation `6.212227`까지 확인했지만 tree residual 파이프라인보다 약했다.

다음 실험은 직접 target을 그대로 예측하는 대신 안정적인 seasonal anchor의 residual만 PatchTST로 학습한다.

```text
prediction =
  Seasonal Naive
  + lambda * PatchTST residual
```

구현 흐름:

1. multivariate input과 cyclic 시간 feature를 사용한다.
2. all-hour origin으로 residual pretrain을 수행한다.
3. midnight origin으로 residual finetune을 수행한다.
4. validation에서 residual lambda를 선택한다.
5. 시간순 validation segment 중 최소 `2/3`에서 개선된 lambda만 허용한다.
6. 안정적인 residual 후보가 없으면 `lambda=0`, 즉 Seasonal Naive로 fallback한다.
7. 선택된 lookback과 seed를 pre-test 전체 label로 refit한다.
8. 최종 제출과 비교용 seasonal anchor를 `multimodel_output/`에 저장한다.

기본 실행은 PatchTST residual 경로다.

```bash
python ett_multimodel_kaggle.py
```

이전 직접 멀티모델 비교를 다시 실행할 때만 다음 옵션을 사용한다.

```bash
python ett_multimodel_kaggle.py --legacy-multimodel
```

첫 학습 확인은 다음 quick 실행으로 진행한다.

```bash
python ett_multimodel_kaggle.py \
  --quick \
  --output multimodel_output/submit_patchtst_residual_quick.csv \
  --report-output multimodel_output/run_report_patchtst_residual_quick.json
```

quick 실행 후 공유할 파일:

```text
multimodel_output/run_report_patchtst_residual_quick.json
```

---

## 14. PatchTST residual quick smoke 실행

### 14.1 실행 명령

RTX GPU가 있는 로컬 `.venv`에서 다음 명령을 실행했다.

```bash
python ett_multimodel_kaggle.py \
  --quick \
  --patch-seeds 42 \
  --patch-lookbacks 336 \
  --output multimodel_output/submit_patchtst_residual_quick.csv \
  --report-output multimodel_output/run_report_patchtst_residual_quick.json
```

`--quick`은 경로 검증용이므로 pretrain과 finetune을 각각 `1` epoch만 실행한다.

### 14.2 validation 결과

```text
Seasonal Naive:
  MSE = 6.874173
  MAE = 2.033128

PatchTST residual, lambda = 1.00:
  MSE = 9.031792
  MAE = 2.315975

Selected PatchTST residual:
  lambda = 0.30
  MSE = 6.425911
  MAE = 1.931638
  positive validation segments = 3/3
```

1 epoch smoke 실행에서도 작은 lambda의 residual 보정은 Seasonal Naive보다 validation이 좋았다. 시간순 validation segment `3/3`에서도 개선되어 안정성 gate를 통과했다.

### 14.3 final refit 검증

validation 단계:

```text
all-hour pretrain samples = 10,753
midnight finetune samples = 449
residual scale = 4.202030
scaler fit end exclusive = 2017-10-10 00:00:00
```

final refit 단계:

```text
all-hour pretrain samples = 13,489
midnight finetune samples = 563
residual scale = 3.941815
scaler fit end exclusive = 2018-02-01 00:00:00
```

기존 absolute direct forecast가 버리던 최신 허용 label 구간을 final refit이 사용한다. test 미래 target은 참조하지 않는다.

### 14.4 판단

smoke 실행은 코드 경로 검증용이므로 Kaggle 제출 대상으로 사용하지 않는다.

다음 full 실행:

```bash
python ett_multimodel_kaggle.py \
  --patch-seeds 42,2024,2026 \
  --patch-lookbacks 336
```

`336` full 실행 결과를 확인한 뒤 계산 비용을 감수할 근거가 있으면 다음 비교를 수행한다.

```bash
python ett_multimodel_kaggle.py \
  --patch-seeds 42,2024,2026 \
  --patch-lookbacks 336,512
```

---

## 15. PatchTST residual 336 full 실행

### 15.1 실행 명령

```bash
python ett_multimodel_kaggle.py \
  --patch-seeds 42,2024,2026 \
  --patch-lookbacks 336
```

실행 설정:

```text
input = multivariate raw 7개 + cyclic time feature 6개
PatchTST lookback = 336
PatchTST seeds = 42, 2024, 2026
all-hour pretrain epochs = 8
midnight finetune epochs = 최대 20
patience = 5
validation segment gate = 최소 2/3 개선
final refit = True
```

### 15.2 validation 결과

| 후보 | MSE | MAE |
| --- | ---: | ---: |
| Seasonal Naive | `6.874173` | `2.033128` |
| PatchTST residual, lambda `1.00` | `9.634416` | `2.423022` |
| PatchTST residual, selected lambda `0.30` | `6.428596` | `1.954536` |

Seasonal Naive 대비 selected 후보 개선:

```text
MSE 감소 = 0.445577
MSE 감소율 = 약 6.48%
```

### 15.3 lambda와 시간순 segment 안정성

validation MSE 최저 후보:

| lambda | 전체 MSE | positive segments | segment 1 improvement | segment 2 improvement | segment 3 improvement |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.30` | `6.428596` | `2/3` | `-0.066781` | `+0.830053` | `+0.573460` |
| `0.20` | `6.455821` | `3/3` | `+0.122584` | `+0.628647` | `+0.503824` |
| `0.15` | `6.514922` | `3/3` | `+0.154602` | `+0.499714` | `+0.423437` |
| `0.10` | `6.604347` | `3/3` | `+0.144844` | `+0.351962` | `+0.312671` |

해석:

1. validation 최저점은 `lambda=0.30`이다.
2. `lambda=0.30`은 segment 1을 소폭 악화시킨다.
3. `lambda=0.20`은 전체 MSE가 `0.027226` 높지만 `3/3` segment를 모두 개선한다.
4. public 일반화를 우선하면 `lambda=0.20`을 보수적 비교 후보로 유지할 가치가 있다.

### 15.4 seed별 finetune 결과

| seed | best finetune epoch | best validation MSE at lambda `1.00` | early stopping |
| ---: | ---: | ---: | --- |
| `42` | `5` | `11.601313` | Yes |
| `2024` | `1` | `10.998046` | Yes |
| `2026` | `3` | `10.896558` | Yes |

해석:

- 모든 seed가 finetune 초반에 최저점을 찍었다.
- residual을 그대로 더하면 과보정된다.
- seed 평균과 작은 lambda shrink가 일반화에 중요하다.
- 현재는 모델 크기나 epoch를 늘리기보다 rolling 안정성을 확인하는 것이 우선이다.

### 15.5 final refit

validation 단계:

```text
scaler fit end exclusive = 2017-10-10 00:00:00
all-hour pretrain samples = 10,753
midnight finetune samples = 449
residual scale = 4.202030
```

final refit 단계:

```text
scaler fit end exclusive = 2018-02-01 00:00:00
all-hour pretrain samples = 13,489
midnight finetune samples = 563
residual scale = 3.941815
```

final refit은 `2018-01-31 23:00:00`까지 허용된 label을 사용한다. test 미래 target은 참조하지 않는다.

### 15.6 제출 분포

| 파일 | mean | std | min | max |
| --- | ---: | ---: | ---: | ---: |
| `submit_seasonal_naive.csv` | `7.642899` | `3.513852` | `-0.915000` | `17.165001` |
| `submit_patchtst_residual.csv` | `7.998451` | `3.434772` | `-0.542008` | `17.404037` |
| 기존 tree `submit_lightgbm_residual.csv` | `7.303621` | `2.868729` | `-0.128225` | `15.075986` |

PatchTST residual과 Seasonal Naive의 차이:

```text
mean shift = +0.355552
MAE difference = 0.640019
RMSE difference = 0.799056
```

이전 absolute direct ensemble 제출 평균 `10.234`보다 훨씬 보수적이다. 하지만 기존 tree residual 후보보다 평균과 분산이 높으므로 Kaggle public score를 연결해 확인해야 한다.

### 15.7 산출물

```text
multimodel_output/submit.csv
multimodel_output/submit_patchtst_residual.csv
multimodel_output/submit_seasonal_naive.csv
multimodel_output/run_report.json
```

selected 제출 SHA-256:

```text
8c27846f8acb67d57ab6b7496b84d9b53a6ed73ad52990e15de9a3f63d37e1c8
```

### 15.8 판단

PatchTST residual 경로는 이전 absolute direct ensemble보다 안정적으로 개선됐다.

하지만 아직 rolling backtest가 없다.

다음 우선순위:

1. 현재 full refit residual로 `lambda=0.20` 보수적 제출 후보를 추가한다.
2. `lambda=0.20`, `0.30`, Seasonal Naive를 rolling backtest로 비교한다.
3. rolling에서 안정성이 확인된 뒤에만 `336,512` lookback blend를 실행한다.
4. 모델 크기 확대와 epoch 증가는 보류한다.

### 15.9 lambda `0.20` 보수적 제출 후보

full refit 학습을 다시 실행하지 않고 동일 residual을 더 보수적으로 shrink한 제출 파일을 생성했다.

```text
multimodel_output/submit_patchtst_residual_lambda020.csv
```

계산:

```text
residual =
  (submit_patchtst_residual - submit_seasonal_naive) / 0.30

lambda020 submission =
  submit_seasonal_naive + 0.20 * residual
```

분포:

```text
mean = 7.879934
std = 3.444856
min = -0.666339
max = 17.207025
```

SHA-256:

```text
062e770612bb7f2067a94e22a9227742057df824acae9a0fc59faff24f75fc1e
```

Kaggle 제출 횟수가 제한적이면 `lambda=0.20` 보수적 후보를 먼저 확인한다. public score가 유효하게 개선될 때 `lambda=0.30` 후보를 추가 비교한다.

## 16. Public `9.19475` 분석과 strong anchor PatchTST v3

### 16.1 public score 해석

사용자가 공유한 PatchTST residual 제출의 Kaggle public MSE:

```text
9.19475
```

직전 안내 흐름상 `lambda=0.20` 보수적 후보를 우선 제출한 것으로 추정한다. 제출 파일을 정확히 연결하려면 Kaggle에 업로드한 파일의 SHA-256을 함께 확인한다.

해석:

1. 이전 absolute direct 멀티모델 public `13`점대보다 유효하게 개선됐다.
2. validation `6.455821`에 비해 public `9.19475`가 높아 시점 이동에 대한 일반화 여유가 부족하다.
3. 기존 leakage-safe baseline anchor public `7.17243`보다도 낮은 성능이므로 PatchTST를 더 크게 만드는 것이 우선은 아니다.
4. 단순 `24h` 반복 Seasonal Naive를 anchor로 사용한 것이 병목이다.
5. PatchTST는 유지하되 더 강하고 보수적인 anchor가 놓친 residual만 학습해야 한다.

### 16.2 코드 변경

`ett_multimodel_kaggle.py` 기본 경로를 다음처럼 변경했다.

```text
strong three-way anchor
  + lambda * PatchTST residual
```

strong anchor 구성:

```text
1. last value
2. previous 96 hours
3. previous week same time
4. horizon별 24시간 block weight
5. horizon별 4시간 block weight
6. 0.75 * fine weight + 0.25 * coarse weight
```

추가 변경:

1. anchor weight는 validation 이전 label만 사용하여 fit한다.
2. final anchor refit도 `2018-01-31 23:00:00` 이전에 끝나는 허용 label만 사용한다.
3. PatchTST residual dataset은 외부에서 만든 leakage-safe anchor prediction을 입력받는다.
4. 기본 stability gate를 `2/3`에서 `3/3` chronological validation segment 개선으로 강화했다.
5. 안정적인 residual이 없으면 `lambda=0`, 즉 strong anchor로 fallback한다.
6. `--anchor seasonal-naive` 옵션으로 기존 단순 anchor 비교를 유지한다.
7. custom `--output` quick 실행이 full-run 보조 파일을 덮어쓰지 않도록 derived output 이름을 분리했다.

### 16.3 strong anchor quick smoke 결과

실행:

```bash
.venv/bin/python ett_multimodel_kaggle.py \
  --quick \
  --patch-seeds 42 \
  --patch-lookbacks 336 \
  --output multimodel_output/submit_patchtst_strong_anchor_quick_v2.csv \
  --report-output multimodel_output/run_report_patchtst_strong_anchor_quick_v2.json
```

validation:

| predictor | MSE | MAE |
| --- | ---: | ---: |
| Seasonal Naive `24h` repeat | `6.874173` | `2.033128` |
| strong three-way anchor | `5.141880` | `1.728620` |
| PatchTST residual, lambda `1.00` | `7.846684` | `2.131153` |
| selected quick predictor | `5.141880` | `1.728620` |

quick 실행에서는 `1` epoch PatchTST residual이 segment `0/3` 개선에 그쳐 `lambda=0` strong anchor fallback이 선택됐다.

이 결과는 full 학습 성능 판단용이 아니다. 다만 strong anchor 교체와 보수적 fallback 경로가 정상 동작하고, anchor 자체가 동일 validation에서 Seasonal Naive보다 크게 개선되는 것을 확인했다.

### 16.4 다음 full 실행

```bash
.venv/bin/python ett_multimodel_kaggle.py \
  --patch-seeds 42,2024,2026 \
  --patch-lookbacks 336 \
  --output multimodel_output/submit.csv \
  --report-output multimodel_output/run_report.json
```

full 실행 후 확인 순서:

1. `multimodel_output/run_report.json`의 strong anchor validation MSE를 확인한다.
2. 선택된 residual `lambda`가 validation segment `3/3`을 모두 개선했는지 확인한다.
3. `lambda=0`이면 `multimodel_output/submit_anchor.csv`와 최종 제출이 동일한지 확인한다.
4. positive segment가 `3/3`인 작은 lambda가 선택될 때만 PatchTST residual 제출을 우선 검토한다.
5. `336,512` lookback blend와 모델 크기 확대는 이 결과를 확인한 뒤 진행한다.

## 17. Anchor 강화 1단계: constrained candidate expansion

### 17.1 목적

PatchTST를 실질적으로 반영하면서 public MSE `6.0` 이하를 목표로 한다.

다만 public score를 직접 튜닝하지 않는다. 먼저 PatchTST가 학습할 residual을 단순하게 만들기 위해 leakage-safe anchor 후보를 제한적으로 확장한다.

이번 단계에서는 rolling backtest를 아직 수행하지 않는다. 따라서 새 후보는 provisional 상태로만 기록하고 기존 PatchTST 기본 anchor를 자동 교체하지 않는다.

### 17.2 추가한 후보

과적합 위험을 억제하기 위해 후보 수와 horizon 자유도를 제한했다.

```text
기존 previous-week three-way:
  24h block
  12h block shrink50
  4h block shrink50
  4h block shrink75

robust level-adjusted week:
  previous_week + clipped mean(last_24h - previous_week_corresponding_24h)
  24h block
  12h block shrink50

previous-week + previous-2weeks blend:
  global_weight * previous_week
  + (1 - global_weight) * previous_2weeks
  24h block
  12h block shrink50
```

level adjustment clip과 전주·2주 전 global weight는 validation 이전 train label만 사용하여 fit한다.

### 17.3 분석 전용 실행

```bash
python ett_multimodel_kaggle.py \
  --anchor-analysis-only \
  --output multimodel_output/submit_anchor_stage1_provisional.csv \
  --report-output multimodel_output/run_report_anchor_stage1.json
```

이 모드는 PatchTST를 학습하지 않는다.

선택 규칙:

```text
1. validation MSE 최저 후보를 찾는다.
2. 최저점 대비 0.5% 이내 후보만 남긴다.
3. 남은 후보 중 complexity가 가장 낮은 recipe를 provisional 후보로 선택한다.
```

### 17.4 validation 결과

| Anchor candidate | Complexity | MSE | MAE |
| --- | ---: | ---: | ---: |
| `threeway_week2blend_12block_shrink50` | `3` | `4.945705` | `1.717240` |
| `threeway_week2blend_24block` | `2` | `4.958831` | `1.723357` |
| `threeway_prevweek_4block_shrink75` | `4` | `5.145918` | `1.729246` |
| `threeway_prevweek_4block_shrink50` | `3` | `5.150795` | `1.731014` |
| `threeway_prevweek_12block_shrink50` | `2` | `5.155705` | `1.732516` |
| `threeway_prevweek_24block` | `1` | `5.172308` | `1.737027` |
| `threeway_levelweek_12block_shrink50` | `3` | `5.378583` | `1.782097` |
| `threeway_levelweek_24block` | `2` | `5.382999` | `1.784680` |

provisional 선택:

```text
threeway_week2blend_24block
validation MSE = 4.958831
validation MAE = 1.723357
complexity = 2
```

fitted global previous-week weight:

```text
validation-stage fit = 0.565069
final refit = 0.564235
```

전주와 2주 전 패턴을 거의 `56:44`로 섞는다.

### 17.5 해석

1. `previous-week + previous-2weeks` global blend는 기존 anchor보다 validation MSE를 낮췄다.
2. `12h` block 후보가 최저점이지만 단순한 `24h` block 후보와 차이는 약 `0.27%`다.
3. 복잡도를 줄이기 위해 `threeway_week2blend_24block`을 provisional 후보로 유지한다.
4. level-adjusted week는 이번 holdout에서 악화되어 현재 승격 대상이 아니다.
5. single holdout 결과만으로 기본 anchor를 교체하지 않는다.

### 17.6 산출물

```text
multimodel_output/submit_anchor_stage1_provisional.csv
multimodel_output/run_report_anchor_stage1.json
```

provisional 제출 SHA-256:

```text
d17b58416e000854a58e9e6c6d1b0d1348be028d7722e81b47325f27cdb5b507
```

### 17.7 검증

기존 default PatchTST quick 경로를 다시 실행했다.

```bash
.venv/bin/python ett_multimodel_kaggle.py \
  --quick \
  --patch-seeds 42 \
  --patch-lookbacks 336 \
  --output multimodel_output/submit_stage1_default_patch_quick.csv \
  --report-output multimodel_output/run_report_stage1_default_patch_quick.json
```

결과:

```text
torch device = cuda
prediction shape = (142, 96)
submit schema assert = passed
submit row assert = passed
```

도구 셸의 `python`에는 PyTorch가 설치되어 있지 않아 PatchTST 실행이 실패했다. conda 환경을 사용할 때는 현재 활성 환경에서 다음 명령으로 PyTorch와 CUDA를 확인한다.

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 17.8 다음 단계

다음 구현은 anchor 강화 2단계 rolling backtest다.

비교 대상:

```text
기존 threeway_prevweek_4block_shrink75
provisional threeway_week2blend_24block
최저점 threeway_week2blend_12block_shrink50
```

fold 평균 MSE, 최악 fold MSE, 개선 fold 수를 비교한 뒤에만 새 anchor를 PatchTST residual 학습 기반으로 승격한다.

## 18. Anchor 강화 2단계: rolling backtest와 기본 anchor 승격

### 18.1 구현

anchor 후보를 single holdout만으로 선택하지 않도록 rolling backtest 모드를 추가했다.

```bash
python ett_multimodel_kaggle.py \
  --anchor-rolling-backtest \
  --output multimodel_output/submit_anchor_stage2_rolling.csv \
  --report-output multimodel_output/run_report_anchor_stage2_rolling.json
```

비교 후보:

```text
legacy baseline:
  threeway_prevweek_4block_shrink75

stage1 provisional:
  threeway_week2blend_24block

stage1 holdout minimum:
  threeway_week2blend_12block_shrink50
```

rolling 규칙:

```text
folds = 4
fold spacing = 45 days
validation window = 111 days
label purge = 96 hours
```

각 fold는 validation 시작 `96시간` 전까지만 anchor weight fit에 사용한다. validation label과 test 미래 target은 fit에 사용하지 않는다.

승격 gate:

```text
1. mean MSE가 legacy baseline보다 낮아야 한다.
2. 최소 3/4 fold에서 legacy baseline을 개선해야 한다.
3. worst-fold MSE가 legacy baseline worst-fold의 102%를 넘지 않아야 한다.
```

### 18.2 rolling 결과

| Anchor recipe | Mean MSE | Worst-fold MSE | Mean MAE | Positive folds |
| --- | ---: | ---: | ---: | ---: |
| `threeway_week2blend_12block_shrink50` | `6.791933` | `7.949320` | `2.019850` | `4/4` |
| `threeway_week2blend_24block` | `6.810774` | `7.962981` | `2.026131` | `4/4` |
| legacy `threeway_prevweek_4block_shrink75` | `6.977999` | `8.060026` | `2.031377` | `0/4` |

fold별 `threeway_week2blend_12block_shrink50` 개선:

| Fold | Legacy MSE | New MSE | Improvement |
| ---: | ---: | ---: | ---: |
| `1` | `8.060026` | `7.949320` | `+0.110706` |
| `2` | `7.949246` | `7.786672` | `+0.162574` |
| `3` | `6.756807` | `6.486034` | `+0.270773` |
| `4` | `5.145918` | `4.945705` | `+0.200213` |

해석:

1. 전주·2주 전 blend는 두 후보 모두 `4/4` fold에서 legacy anchor를 개선했다.
2. `12h shrink50` 후보가 평균 MSE와 최악 fold MSE 모두 가장 낮다.
3. single holdout 우연이 아니라 여러 과거 구간에서 반복적으로 개선됐다.
4. 보수적 rolling gate를 통과했으므로 `threeway_week2blend_12block_shrink50`를 PatchTST residual 기본 anchor로 승격했다.

### 18.3 승격 anchor 산출물

```text
multimodel_output/submit_anchor_stage2_rolling.csv
multimodel_output/run_report_anchor_stage2_rolling.json
```

SHA-256:

```text
e43c67b148a244828e749f5dcbd973042fcfa6e19ce2e2138f6913d0a66bc2d4
```

제출 분포:

```text
mean = 7.331858
std = 2.940667
min = 0.248749
max = 14.865847
```

### 18.4 PatchTST 연결 smoke

승격 anchor를 PatchTST residual 기본 경로에 연결하고 GPU quick smoke를 실행했다.

```bash
.venv/bin/python ett_multimodel_kaggle.py \
  --quick \
  --patch-seeds 42 \
  --patch-lookbacks 336 \
  --output multimodel_output/submit_stage2_promoted_patch_quick.csv \
  --report-output multimodel_output/run_report_stage2_promoted_patch_quick.json
```

결과:

```text
torch device = cuda
anchor = threeway_week2blend_12block_shrink50
anchor validation MSE = 4.945705
residual scale = 3.459270
prediction shape = (142, 96)
submit schema assert = passed
submit row assert = passed
```

quick `1` epoch PatchTST residual은 안정성 gate를 통과하지 못해 `lambda=0` fallback됐다. 이는 smoke 경로 검증 결과이며 full 학습 성능 판단은 아니다.

### 18.5 다음 full 실행

```bash
python ett_multimodel_kaggle.py \
  --patch-seeds 42,2024,2026 \
  --patch-lookbacks 336 \
  --output multimodel_output/submit.csv \
  --report-output multimodel_output/run_report.json
```

다음 분석에서는 강화된 anchor 위에서 PatchTST residual이 validation segment `3/3`을 개선하는지 확인한다.

## 19. PatchTST 일반화 강화: nested rolling 진단과 RevIN-style 입력 정규화

### 19.1 문제 확인

승격된 anchor 위에서 PatchTST residual이 특정 holdout에만 맞지 않는지 확인하기 위해 nested rolling backtest 모드를 추가했다.

```bash
python ett_multimodel_kaggle.py \
  --patch-residual-rolling-backtest \
  --rolling-patch-seed 42 \
  --rolling-pretrain-epochs 3 \
  --rolling-finetune-epochs 8 \
  --rolling-patience 3 \
  --report-output multimodel_output/run_report_patch_residual_rolling.json
```

rolling 규칙:

```text
outer folds = 4
outer spacing = 45 days
outer validation window = 45 days
inner validation window = 45 days
label purge = 96 hours
fixed lambda candidates = 0.00, 0.05, 0.10, 0.15
```

각 fold의 scaler와 anchor는 inner validation 이전 학습 구간에만 fit한다. early stopping과 lambda 선택도 inner validation만 사용하고, outer label은 일반화 진단 결과를 출력할 때만 참조한다.

기존 PatchTST residual 결과:

| Lambda | Mean MSE | Worst-fold MSE | Mean improvement | Positive folds |
| ---: | ---: | ---: | ---: | ---: |
| `0.00` | `7.439213` | `9.390017` | `0.000000` | `0/4` |
| `0.05` | `7.427274` | `9.427492` | `+0.011939` | `2/4` |
| `0.10` | `7.471732` | `9.506469` | `-0.032519` | `2/4` |
| `0.15` | `7.572588` | `9.626949` | `-0.133375` | `2/4` |

해석:

1. 기존 PatchTST residual은 일부 기간에서는 anchor를 개선하지만 최근 fold에서는 악화됐다.
2. 단순 lambda shrink만으로는 일반화 문제를 해결하기 어렵다.
3. 시계열 level shift에 대한 PatchTST 입력 민감도를 낮추는 방향이 필요하다.

### 19.2 RevIN-style 입력 정규화 구현

PatchTST 수치 입력 채널에 window 단위 정규화를 추가했다.

```text
1. 각 입력 window의 수치 채널별 mean/std를 계산한다.
2. 수치 입력을 window 내부 기준으로 정규화한다.
3. cyclic 시간 feature는 기존 값을 유지한다.
4. 원래 level 정보를 잃지 않도록 mean/std를 context feature로 함께 전달한다.
5. residual 출력 구조는 유지한다.
```

멀티변수 입력에서는 `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`를 정규화한다. 이 옵션은 기본 활성화되며 비교 실험이 필요하면 `--no-patch-window-revin`으로 끌 수 있다.

### 19.3 RevIN-style nested rolling 결과

```bash
python ett_multimodel_kaggle.py \
  --patch-residual-rolling-backtest \
  --patch-window-revin \
  --rolling-patch-seed 42 \
  --rolling-pretrain-epochs 3 \
  --rolling-finetune-epochs 8 \
  --rolling-patience 3 \
  --report-output multimodel_output/run_report_patch_residual_rolling_revin.json
```

| Lambda | Mean MSE | Worst-fold MSE | Mean improvement | Positive folds |
| ---: | ---: | ---: | ---: | ---: |
| `0.00` | `7.439213` | `9.390017` | `0.000000` | `0/4` |
| `0.05` | `7.259731` | `9.189315` | `+0.179482` | `4/4` |
| `0.10` | `7.153663` | `9.066660` | `+0.285550` | `4/4` |
| `0.15` | `7.121010` | `9.022052` | `+0.318203` | `4/4` |

`lambda=0.15`의 fold별 개선:

| Fold | Improvement |
| ---: | ---: |
| `1` | `+0.243564` |
| `2` | `+0.472280` |
| `3` | `+0.367965` |
| `4` | `+0.189005` |

해석:

1. RevIN-style 입력 정규화 적용 후 `0.05`, `0.10`, `0.15`가 모두 rolling gate를 통과했다.
2. `lambda=0.15`는 평균 MSE를 `7.439213`에서 `7.121010`으로 낮췄다.
3. 네 fold를 모두 개선했으므로 단일 holdout 우연보다 일반화 개선 근거가 강하다.
4. 최종 제출 lambda는 rolling 결과만 고정하지 않고 pre-test holdout segment gate로 보수적으로 선택한다.

### 19.4 PatchTST 3-seed full 실행

```bash
python ett_multimodel_kaggle.py \
  --patch-seeds 42,2024,2026 \
  --patch-lookbacks 336 \
  --output multimodel_output/submit.csv \
  --report-output multimodel_output/run_report.json
```

검증 결과:

| Candidate | MSE | MAE |
| --- | ---: | ---: |
| Seasonal Naive | `6.874173` | `2.033128` |
| 승격 anchor | `4.945705` | `1.717240` |
| PatchTST residual `lambda=1.00` | `12.569520` | `2.747048` |
| 선택 PatchTST residual `lambda=0.05` | `4.833202` | `1.708968` |

`lambda=0.05`의 시간순 validation segment 개선:

| Segment | Anchor MSE | Candidate MSE | Improvement |
| ---: | ---: | ---: | ---: |
| `1` | `5.076160` | `5.074402` | `+0.001757` |
| `2` | `3.457306` | `3.267584` | `+0.189722` |
| `3` | `6.303649` | `6.157619` | `+0.146030` |

holdout 전체 MSE는 anchor 대비 `0.112503`, 약 `2.28%` 개선됐다. `lambda=0.10`, `0.15`가 holdout 평균 MSE만 보면 더 낮았지만 첫 번째 validation segment를 악화시켰다. 따라서 segment `3/3` 안정성 gate를 통과한 `lambda=0.05`를 최종 제출 후보로 선택했다.

### 19.5 산출물

```text
multimodel_output/submit.csv
multimodel_output/submit_patchtst_residual.csv
multimodel_output/run_report.json
multimodel_output/run_report_patch_residual_rolling.json
multimodel_output/run_report_patch_residual_rolling_revin.json
```

최종 `submit.csv` SHA-256:

```text
af25d9597d4279bfb8f1b6b1c7d97f11bd6c818c684576b9122946c79810544b
```

제출 분포:

```text
mean = 7.375609
std = 2.946607
min = 0.337845
max = 14.887825
```

### 19.6 판단

이번 개선은 public score를 기준으로 튜닝하지 않았다. 누수 없는 nested rolling `4/4` fold 개선과 pre-test holdout segment `3/3` 개선을 모두 통과한 보수적 PatchTST residual 후보다.

다음 Kaggle 제출 대상은 `multimodel_output/submit.csv`다. public score는 외부 확인 지표로만 기록하고, 다음 코드 변경도 rolling 및 holdout 일반화 근거가 있을 때만 진행한다.

### 19.7 Kaggle public 결과와 다음 단계

Kaggle public MSE:

```text
6.49891
```

비교:

| 제출 후보 | Public MSE |
| --- | ---: |
| 기존 비-PatchTST 최고 후보 | `6.62817` |
| RevIN-style PatchTST residual | `6.49891` |
| 개선 | `+0.12926` |

해석:

1. PatchTST residual이 실제 Kaggle public 구간에서도 기존 비-PatchTST 최고 후보를 개선했다.
2. public score는 개선 방향을 확인하는 외부 지표로만 사용한다.
3. 다음 실험에서도 public leaderboard에 맞춰 lambda를 직접 조절하지 않는다.

다음 단계는 PatchTST lookback 다양화다.

```text
현재 기준: PatchTSTResidual-336
추가 후보: PatchTSTResidual-512
추가 후보: PatchTSTResidual-blend-336-512
```

선정 규칙:

```text
1. 336h, 512h, 336h+512h residual blend를 동일 split에서 비교한다.
2. nested rolling backtest에서 평균 MSE, worst-fold MSE, positive fold 수를 확인한다.
3. pre-test holdout segment gate를 통과한 후보만 제출 대상으로 허용한다.
4. 새 후보가 rolling 및 holdout 기준을 개선하지 못하면 현재 336h 제출을 유지한다.
```
