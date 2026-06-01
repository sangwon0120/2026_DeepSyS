# ETTh1 Forecasting 모델 실험 및 개선 이력

## 1. 문서 목적

이 문서는 `2026_DeepSyS` ETTh1 예측 프로젝트에서 진행한 대화, 실험 결과, 판단 근거, 코드 변경 사항을 시간 순서대로 기록한다.

핵심 목적은 단순히 "어떤 모델을 사용했다"를 나열하는 것이 아니다. 각 단계에서 다음 내용을 추적할 수 있도록 작성했다.

1. 이전 모델의 결과가 어땠는가.
2. 그 결과에서 어떤 문제를 발견했는가.
3. 문제를 해결하기 위해 어떤 변경을 적용했는가.
4. 변경 후 validation, rolling backtest, public score가 어떻게 변했는가.
5. 결과를 근거로 다음 개선 방향을 어떻게 결정했는가.

새 대화 세션에서 작업을 이어갈 때는 이 문서를 먼저 읽고, 가장 최근의 실험 상태와 보류된 판단을 확인하면 된다.

---

## 2. 문제 정의와 점수 해석

### 2.1 데이터셋

사용 데이터는 `ETTh1.csv`다.

- timestamp 수: `17,420`
- 주기: hourly
- 변수 수: `7`
- 변수:
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

### 2.2 이 프로젝트의 MSE는 raw OT scale

이 프로젝트의 validation 및 Kaggle public score는 정규화된 MSE가 아니라 `OT` 원 단위의 raw MSE다.

train 구간 기준 통계:

```text
train_end_idx = 10440
train split boundary = 2017-09-09 00:00:00
OT mean = 17.2959388
OT std = 8.5183143
```

따라서 ETTh1 benchmark 문서에서 흔히 보는 normalized MSE `0.30~0.45`와 이 프로젝트의 raw MSE `5~7`을 직접 비교하면 안 된다.

예시 환산:

| 항목 | raw MSE | train OT std 기준 normalized MSE 환산 |
| --- | ---: | ---: |
| seasonal baseline validation | `5.838997` | 약 `0.0805` |
| CatBoost validation 후보 | `5.658024` | 약 `0.0780` |
| 제출 public score | `6.62817` | 약 `0.0913` |

normalized MSE가 일반 benchmark보다 낮아 보이는 이유는 이 프로젝트의 validation/test가 일반 ETTh1 benchmark split과 동일하지 않기 때문이다. 이 프로젝트는 자정 origin 중심으로 96시간을 예측한다.

### 2.3 현재 평가 원칙

초기에는 public score 목표 `6.00`을 강하게 의식했다. 이후 대화를 통해 다음 원칙으로 수정했다.

1. 모델 선택의 1차 기준은 validation raw MSE다.
2. rolling backtest는 최종 선택을 강제로 막는 gate가 아니라 시점 이동 안정성을 확인하는 진단 정보다.
3. public score는 validation 방향이 실제 제출에서도 유지되는지 확인하는 보조 신호다.
4. public score 하나에 과적합하지 않는다.
5. validation이 내려가더라도 rolling 결과가 크게 무너지면 별도 안전 후보를 함께 생성한다.

현재 현실적인 validation 목표:

| 단계 | validation raw MSE 목표 |
| --- | ---: |
| 현재 seasonal baseline | `5.84` 수준 |
| 현재 tree residual 후보 | `5.66~5.72` 수준 |
| 다음 1차 목표 | `5.50` 이하 |
| 강한 개선 목표 | `5.30` 이하 |
| 공격적 목표 | `5.00` 이하 |

---

## 3. 파일 구조

### 3.1 기존 baseline 및 딥러닝 실험

파일: `ett_forecasting_pytorch.py`

주요 역할:

- 데이터 로딩
- seasonal baseline 후보 생성 및 평가
- rolling baseline backtest
- GRU + attention residual booster
- custom PatchTST residual booster
- 제출 파일 생성

### 3.2 tree residual 앙상블 실험

파일: `ett_ensemble_pipeline.py`

주요 역할:

- 기존 seasonal baseline 재사용
- 누락값/이상값 처리
- all-hour tabular residual 학습 데이터 생성
- LightGBM residual 학습
- CatBoost residual 학습
- CatBoost seed ensemble
- residual lambda grid search
- weighted 후보 생성
- validation MSE 최저 후보 선택
- rolling backtest 후보별 진단
- 후보별 제출 파일 생성

### 3.3 로컬 실행 wrapper

파일: `run_local_ensemble.py`

로컬 환경 기준 설정:

- Ryzen 7500F CPU를 고려한 thread 수 설정
- 기본값: logical thread 중 2개를 남기고 사용
- 권장 실행값: `--threads 10`
- RTX 4060 Ti CatBoost GPU 시도 옵션: `--catboost-gpu`
- GPU CatBoost 실패 시 CPU fallback
- 결과 CSV 자동 저장

---

## 4. 초기 seasonal baseline 실험

### 4.1 관찰

가장 먼저 단순 seasonal 계열 baseline을 비교했다.

public score 기록:

| 제출 방식 | public MSE |
| --- | ---: |
| `last_value` | `8.33911` |
| `blend_last_value_last_week` | `7.30652` |
| `threeway_last_prev_week_4block` | `7.21015` |
| `threeway_last_prev_week_24block_shrink75` | `7.17243` |

local validation에서 seasonal 후보를 상세 비교한 결과, 수정 후 안정적으로 가장 좋은 후보는 다음이었다.

```text
threeway_last_prev_week_24block_shrink75
validation MSE = 5.838997
validation RMSE = 2.416402
validation MAE = 1.855697
```

이 baseline은 다음 세 예측을 horizon별 weight로 결합한다.

1. `last_value`
2. `last_96h_repeat`
3. `last_week_same_time`

`24block`은 96개 horizon을 4시간 단위로 세밀하게 나눠 weight를 최적화한다. `shrink75`는 세밀한 weight를 더 거친 24시간 block weight 쪽으로 일부 shrink해서 과적합 위험을 줄인다.

### 4.2 판단

단순 seasonal baseline이 이미 강했다. 따라서 이후 딥러닝이나 boosting 모델은 target 자체를 처음부터 직접 예측하기보다, seasonal baseline이 놓치는 residual만 학습하는 방식이 실패 위험이 낮다고 판단했다.

최종 구조의 기본 원칙:

```text
final_prediction = seasonal_baseline + lambda * residual_prediction
```

### 4.3 중요한 정정

초기에 `threeway_last_prev_week_24block`을 제출했고 public MSE `7.19015`가 나왔다.

기존 best였던 `threeway_last_prev_week_24block_shrink75 = 7.17243`보다 나쁜 점수다. 숫자 비교를 잘못 해석했다가 대화에서 정정했다.

이후 baseline 재현 경로는 `threeway_last_prev_week_24block_shrink75`를 기준으로 유지했다.

---

## 5. 환경 의존성 문제와 최소 수정

### 5.1 문제

초기 `ett_forecasting_pytorch.py`는 top-level에서 `matplotlib`를 import했다. `plot=False`로 학습만 실행하려고 해도 import 단계에서 `ModuleNotFoundError`가 발생했다.

### 5.2 변경

`matplotlib` import를 `plot_history()` 내부로 옮겼다.

### 5.3 판단

모델 구조, 학습 로직, 예측 로직은 건드리지 않았다. 라이브러리가 없는 환경에서 plotting을 사용하지 않을 때 import 자체가 실패하는 문제만 최소 범위로 해결했다.

---

## 6. ResidualMLP에서 GRU + attention residual booster로 발전

### 6.1 이전 구조

기존 residual 보정기는 `ResidualMLP`였다. seasonal baseline이 강하기 때문에 MLP residual 보정을 유지하되 sequence modeling 능력을 강화할 필요가 있다고 판단했다.

### 6.2 적용한 변경

`ResidualMLP`를 `SeqResidualBooster`로 교체했다.

모델 구조:

```text
recent sequence
  -> GRU(hidden=128, layers=2, dropout=0.2)
  -> attention-lite pooling
  -> last hidden + attention context + aux embedding concat
  -> MLP head
  -> 96-step residual
```

초기 예측이 baseline과 동일하도록 마지막 linear layer를 zero initialization했다.

입력:

- 최근 168시간 scaled raw/time features
- daily delta
- weekly delta
- baseline forecast 96개
- future time features
- 24/168/336/672시간 raw 통계
- 최근 seasonal residual 통계

학습 흐름:

- all-hour pretrain
- midnight target finetune
- validation에서 residual lambda `0.00~1.00`, `0.05` 간격 탐색
- baseline보다 나쁘면 fallback

### 6.3 첫 실행 문제

rolling backtest 내부 학습에서 다음 assertion으로 실행이 중단됐다.

```text
AssertionError: validation MSE target not met: 11.93687 >= 10.0
```

### 6.4 판단과 수정

rolling fold는 시기별 난이도가 다르다. 특정 fold가 목표 MSE `10.0`을 넘는다는 이유로 전체 파이프라인을 중단하면 fallback 정책을 평가할 수 없다.

따라서 assertion을 warning으로 바꿨다. 모델이 나쁘면 baseline으로 fallback하도록 선택 로직에 책임을 맡겼다.

### 6.5 Seq residual 결과

한 실행의 main validation:

| 항목 | MSE |
| --- | ---: |
| baseline | `5.862435` |
| best Seq residual, lambda `0.15` | `5.645186` |

main validation만 보면 개선됐다.

하지만 rolling residual backtest는 나빴다.

| fold | outer baseline MSE | outer residual MSE | improvement |
| --- | ---: | ---: | ---: |
| 1 | `7.604089` | `10.090480` | `-2.486391` |
| 2 | `7.183193` | `8.079773` | `-0.896580` |
| 3 | `5.931369` | `5.931014` | `+0.000355` |

요약:

```text
mean improvement = -1.127539 수준
positive folds = 1/3
```

### 6.6 판단

GRU residual은 main validation에 적합했지만 시점 이동에 취약했다. validation 하나만 보고 제출하면 위험했다.

결정:

- Seq residual은 제출 후보에서 제외
- seasonal baseline fallback 유지
- sequence residual 계열은 더 장기 구조인 PatchTST로 한 번 더 검토

---

## 7. baseline split 수정과 재현성 확보

### 7.1 문제

긴 lookback residual 모델을 추가하면서 baseline fitting까지 긴 lookback target으로 제한될 위험이 있었다. 그러면 기존 public best baseline과 동일한 방식이 아니게 된다.

### 7.2 변경

baseline과 residual model의 target 집합을 분리했다.

- baseline fitting:
  - 기존처럼 `LOOKBACK=168`
  - midnight target 사용
- Seq residual:
  - `SEQ_LOOKBACK=672`
- PatchTST residual:
  - 별도 long-lookback target 사용

### 7.3 결과

baseline 재현 결과:

```text
threeway_last_prev_week_24block_shrink75
validation MSE = 5.838997
```

### 7.4 판단

이후 실험은 항상 이 baseline을 anchor로 삼는다. 새로운 모델이 실패해도 기존 제출 품질을 보존한다.

---

## 8. custom PatchTST residual 실험

### 8.1 왜 PatchTST를 검토했는가

GRU residual은 최근 sequence를 보지만 rolling backtest에서 불안정했다. ETTh1 계열 long-horizon benchmark에서 PatchTST 계열이 강하다는 점을 고려해 residual 구조에 patching을 적용했다.

off-the-shelf `neuralforecast.PatchTST`를 그대로 사용하지 않은 이유:

- 우리의 target은 단일 OT direct forecast가 아니다.
- origin별 seasonal baseline forecast에 대한 `96-step residual matrix`다.
- 기존 fallback 및 lambda 선택 흐름을 유지해야 했다.

따라서 custom PyTorch module `PatchTSTResidual`을 `ett_forecasting_pytorch.py`에 구현했다.

### 8.2 전처리

PatchTST 경로에서 추가한 처리:

1. timestamp 정렬
2. 중복 timestamp 제거
3. hourly reindex
4. numeric column별 누락값 처리
   - linear interpolation
   - forward fill
   - backward fill
5. train 구간 기준 IQR clipping threshold fit
6. 전체 feature에 winsorize 적용
7. train 구간 기준 scaling

중요:

- metric 계산용 raw target은 clipping하지 않는다.
- model feature만 clipping한다.
- scaler와 threshold는 train 구간에만 fit한다.

### 8.3 PatchTST 구조

```text
context length = 672
patch length = 24
stride = 12
d_model = 128
n_heads = 8
encoder layers = 3
d_ff = 256
dropout = 0.2
```

입력:

- raw 7개
- time cyclic 8개
- raw daily delta 7개
- raw weekly delta 7개
- baseline forecast
- future time features
- 24/168/336/672 rolling stats
- recent residual stats

### 8.4 결과

main validation:

| 항목 | MSE |
| --- | ---: |
| baseline lambda `0.00` | `5.838997` |
| PatchTST residual lambda `0.05` | `5.825482` |

main validation 개선은 약 `0.23%`로 매우 작았다.

rolling PatchTST residual backtest:

| fold | outer baseline MSE | outer residual MSE | improvement |
| --- | ---: | ---: | ---: |
| 1 | `7.565614` | `12.240747` | `-4.675133` |
| 2 | `7.152225` | `10.417562` | `-3.265337` |
| 3 | `5.916977` | `6.355700` | `-0.438724` |

요약:

```text
baseline mean MSE = 6.878272
PatchTST residual mean MSE = 9.671336
mean improvement = -2.793064
positive folds = 0/3
```

### 8.5 판단

PatchTST residual은 실패했다.

- main validation 개선폭이 너무 작다.
- rolling backtest에서는 모든 fold가 악화됐다.
- 비싼 딥러닝 실험을 더 확대할 근거가 없다.

결정:

- PatchTST 제출 금지
- baseline fallback 유지
- 작은 데이터에서 더 안정적인 tabular boosting residual로 방향 전환

---

## 9. 다양한 앙상블 후보 검토

다음 조합을 검토했다.

- LightGBM
- CatBoost
- N-HiTS
- N-BEATS
- Chronos-Bolt
- Seasonal Naive

### 9.1 판단

모델 다양성 자체는 의미가 있다.

- LightGBM / CatBoost:
  - lag feature
  - rolling mean
  - calendar feature
  - engineered residual feature
- N-HiTS / N-BEATS:
  - target 자체의 시계열 구조
- Chronos-Bolt:
  - pretrained foundation model의 zero-shot 보완
- Seasonal Naive:
  - 안정적인 anchor

하지만 현재 데이터가 작고, 딥러닝 residual이 rolling에서 계속 실패했다. 처음부터 모든 모델을 붙이면 계산 비용과 validation 과적합 위험이 커진다.

따라서 구현 우선순위를 다음처럼 정했다.

1. 기존 seasonal anchor 유지
2. LightGBM residual
3. CatBoost residual
4. validation 및 rolling 결과 확인
5. 필요할 때 N-HiTS, Chronos-Bolt, N-BEATS 검토

현재 구현은 1~4 단계까지다.

---

## 10. 별도 tree residual 파이프라인 생성

새 파일:

```text
ett_ensemble_pipeline.py
```

기존 `ett_forecasting_pytorch.py`를 유지한 이유:

- seasonal baseline 재현 경로 보존
- Seq residual / PatchTST 실험 기록 보존
- 새로운 tree residual 실험을 별도 실행 가능
- Colab/Kaggle과 로컬 실행을 분리 가능

초기 `run_ensemble()` 구조:

```text
load data
  -> clean / clip / robust-scale features
  -> reproduce seasonal anchor
  -> build tabular residual dataset
  -> train LightGBM residual
  -> train CatBoost residual
  -> weighted blend with anchor
  -> rolling backtest
  -> save submit
```

---

## 11. 첫 tree residual 실험: midnight-only 학습

### 11.1 초기 feature

tabular residual feature:

- horizon:
  - horizon index
  - horizon day
  - horizon block
  - sin/cos
- origin calendar
- target calendar
- anchor forecast
- `last_value`
- `last_96h_repeat`
- `last_week_same_time`
- anchor component weight
- OT lag:
  - `1, 2, 3, 6, 12, 24, 48, 72, 96, 168, 336, 672`
- raw feature lag:
  - `1, 24, 168`
- OT rolling stats:
  - `3, 6, 12, 24, 48, 96, 168, 336, 672`
- raw rolling stats:
  - `24, 168`
- trend / delta feature

target:

```text
true_OT - seasonal_anchor_prediction
```

### 11.2 결과

midnight-only 학습 origin 수:

```text
404
```

main validation:

| 후보 | MSE |
| --- | ---: |
| anchor | `5.838997` |
| LightGBM residual | `5.794099` |
| CatBoost residual | `5.566250` |
| weighted ensemble | `5.562830` |

main validation만 보면 매우 좋아 보였다.

하지만 rolling backtest:

| fold | outer baseline MSE | outer ensemble MSE | improvement |
| --- | ---: | ---: | ---: |
| 1 | `7.565614` | `7.617765` | `-0.052151` |
| 2 | `7.152225` | `7.152225` | `0.000000` |
| 3 | `5.916977` | `6.890478` | `-0.973501` |

요약:

```text
baseline mean MSE = 6.878272
ensemble mean MSE = 7.220156
mean improvement = -0.341884
positive folds = 0/3
```

### 11.3 판단

tree residual도 처음에는 main validation에만 맞고 시점 이동에 실패했다.

핵심 원인 후보:

1. midnight origin `404`개만 학습해서 샘플 수가 너무 작다.
2. baseline이 최근에 어떤 방향으로 틀렸는지 알려주는 직접 feature가 없다.
3. residual 적용 강도가 너무 공격적이다.
4. horizon별 residual 분산 차이를 반영하지 않았다.

---

## 12. tree residual 안정화: all-hour, 과거 오차 feature, horizon 정규화

### 12.1 적용한 변경

#### all-hour 학습 origin 확장

validation/test는 midnight target을 유지하고, tree residual 학습 origin만 all-hour로 확장했다.

```text
midnight-only origin = 404
all-hour origin = 9,505
```

#### 과거 baseline error feature

현재 시점에서 이미 관측이 끝난 과거 baseline 예측 오차를 feature로 추가했다.

사용 lag:

```text
t-96
t-168
t-336
t-672
```

각 과거 error curve에서 추가한 정보:

- mean
- std
- absolute mean
- last residual
- sign mean
- 24시간 block mean
- 동일 horizon residual
- 동일 horizon absolute residual

누수 방지 조건:

```text
past_start + HORIZON <= current_target_idx
```

즉 현재 예측 origin 시점에 아직 알 수 없는 미래 오차는 feature에 포함하지 않는다.

#### horizon별 residual 정규화

96개 horizon은 residual 분산이 다르다. 뒤쪽 horizon이 더 어렵기 때문에 raw residual을 그대로 학습하면 특정 horizon에 끌릴 수 있다.

적용:

```text
normalized_residual =
  (residual - horizon_train_mean) / horizon_train_std
```

예측 후 다시 raw residual scale로 복원한다.

#### 보수적인 blend

weighted 후보의 anchor 최소 weight를 높였다.

```text
anchor min weight = 0.70
```

### 12.2 결과

main validation:

| 후보 | MSE |
| --- | ---: |
| anchor | `5.838997` |
| LightGBM residual | `5.781616` |
| CatBoost residual | `5.710577` |
| weighted ensemble | `5.795801` |

rolling backtest:

| fold | outer baseline MSE | outer weighted MSE | improvement |
| --- | ---: | ---: | ---: |
| 1 | `7.565614` | `7.522726` | `+0.042888` |
| 2 | `7.152225` | `6.921857` | `+0.230368` |
| 3 | `5.916977` | `5.870407` | `+0.046570` |

요약:

```text
baseline mean MSE = 6.878272
weighted ensemble mean MSE = 6.771663
mean improvement = +0.106609
positive folds = 3/3
```

### 12.3 판단

all-hour 학습과 leakage-free 과거 error feature가 효과가 있었다.

처음으로 다음 두 조건을 동시에 만족했다.

1. main validation 개선
2. rolling backtest 3/3 개선

하지만 validation 최저 후보는 weighted ensemble이 아니라 CatBoost 단독이었다.

결정:

- 최종 선택을 weighted ensemble으로 강제하지 않는다.
- 후보 중 validation MSE가 가장 낮은 모델을 제출한다.

---

## 13. 선택 정책 변경: validation 최저 후보 선택

### 13.1 이전 정책

초기에는 residual 모델이 validation을 개선해도 rolling gate를 통과하지 못하면 baseline으로 fallback했다.

이 정책은 안전하지만, 사용자가 public score보다 validation score를 중심으로 개선하겠다는 방향을 정한 뒤 변경했다.

### 13.2 현재 정책

다음 후보를 모두 validation table에 올린다.

```text
anchor
lightgbm_residual
catboost_residual
weighted_anchor_lgbm_catboost
```

최종 제출:

```text
validation MSE가 가장 낮은 후보
```

rolling backtest:

```text
선택을 막는 hard gate가 아니라 diagnostic
```

### 13.3 public 결과

CatBoost residual 제출에서 보고된 public score:

```text
public MSE = 6.62817
```

기존 seasonal best:

```text
public MSE = 7.17243
```

개선:

```text
약 7.59% 감소
```

### 13.4 판단

validation 개선 방향이 public에서도 의미 있게 이어졌다.

다만 validation과 public은 1:1로 움직이지 않는다. 이후에도 validation을 1차 기준으로 삼되 public은 후보 필터링용 보조 신호로 기록한다.

---

## 14. 로컬 실행 환경 전환

### 14.1 이유

Colab GPU 한도를 모두 사용했기 때문에 tree residual 앙상블을 로컬에서 실행하기로 했다.

사용 하드웨어:

```text
CPU = Ryzen 7500F
GPU = RTX 4060 Ti
RAM = 32 GB
```

### 14.2 가상환경

기존 `.venv`를 재사용했다.

설치 패키지:

```text
numpy
pandas
torch
lightgbm
catboost
matplotlib
scikit-learn
```

LightGBM 실행 중 다음 오류가 발생해 `scikit-learn`을 추가 설치했다.

```text
lightgbm.basic.LightGBMError:
scikit-learn is required for lightgbm.sklearn
```

### 14.3 로컬 최적화

`run_local_ensemble.py`에 적용한 설정:

- 기본 thread 수:
  - `os.cpu_count() - 2`
- Ryzen 7500F 권장:
  - `--threads 10`
- LightGBM:
  - `n_jobs=threads`
  - `force_col_wise=True`
- CatBoost CPU:
  - `thread_count=threads`
- CatBoost GPU:
  - `--catboost-gpu`
  - 실패 시 CPU fallback
- BLAS 계열 환경 변수 설정:
  - `OMP_NUM_THREADS`
  - `MKL_NUM_THREADS`
  - `OPENBLAS_NUM_THREADS`
  - `NUMEXPR_NUM_THREADS`
- matplotlib cache:
  - 프로젝트 내부 `.matplotlib_cache`

---

## 15. CatBoost seed ensemble과 residual lambda 탐색

### 15.1 개선 이유

CatBoost residual 단독이 weighted ensemble보다 validation이 좋았다.

이전 결과:

```text
CatBoost residual validation MSE = 5.710577
weighted validation MSE = 5.795801
```

따라서 CatBoost residual을 더 안정화하고 residual 적용 강도를 validation으로 선택하기로 했다.

### 15.2 적용한 변경

#### CatBoost seed ensemble

기본 seed:

```text
42
1337
2026
```

각 CatBoost 모델의 residual 예측을 평균한다.

#### residual lambda grid search

초기:

```text
0.50 ~ 1.20
step = 0.05
```

상한 `1.20`에서 validation 최저점이 나와 이후 확장했다.

현재:

```text
0.50 ~ 1.60
step = 0.05
```

#### 후보별 제출 파일

최종 선택 파일 외에 각 후보를 별도 저장하도록 변경했다.

```text
submit_ensemble.csv
submit_anchor.csv
submit_lightgbm_residual.csv
submit_catboost_residual.csv
submit_weighted_anchor_lgbm_catboost.csv
```

`submit_ensemble.csv`는 validation 최저 후보와 동일한 예측이다.

### 15.3 lambda `1.20` 시점 결과

| 후보 | validation MSE | lambda |
| --- | ---: | ---: |
| CatBoost residual | `5.658024` | `1.20` |
| LightGBM residual | `5.733257` | `1.20` |
| weighted | `5.778386` | - |
| anchor | `5.838997` | - |

판단:

- validation은 더 내려갔다.
- CatBoost residual이 다시 가장 좋았다.
- 두 residual 모델 모두 상한 `1.20`에서 최저점이므로 grid 확장이 필요했다.

### 15.4 lambda `1.60` 확장 후 최신 결과

최신 `ensemble_candidates.csv`:

| 후보 | validation MSE | RMSE | MAE | lambda |
| --- | ---: | ---: | ---: | ---: |
| LightGBM residual | `5.713983` | `2.390394` | `1.845855` | `1.60` |
| CatBoost residual | `5.717802` | `2.391192` | `1.845914` | `1.60` |
| weighted | `5.787499` | `2.405722` | `1.850264` | - |
| anchor | `5.838997` | `2.416402` | `1.855697` | - |

최신 main validation 선택:

```text
lightgbm_residual
lambda = 1.60
validation MSE = 5.713983
```

rolling backtest에서 inner validation 최저 후보를 outer에 적용한 결과:

| fold | inner selected | outer baseline MSE | outer selected MSE | improvement |
| --- | --- | ---: | ---: | ---: |
| 1 | CatBoost residual | `7.565614` | `7.607316` | `-0.041702` |
| 2 | CatBoost residual | `7.152225` | `6.719676` | `+0.432549` |
| 3 | CatBoost residual | `5.916977` | `6.234251` | `-0.317275` |

요약:

```text
mean improvement = +0.024524
positive folds = 1/3
selected max MSE = 7.607316
baseline max MSE = 7.565614
```

같은 실행의 weighted outer MSE:

| fold | weighted outer MSE |
| --- | ---: |
| 1 | `7.465306` |
| 2 | `6.929019` |
| 3 | `5.837275` |

### 15.5 판단

최신 결과는 validation 기준으로 anchor보다 좋지만, 이전 CatBoost validation best `5.658024`보다는 나쁘다.

핵심 해석:

1. lambda를 키운다고 항상 개선되지는 않는다.
2. seed ensemble은 분산을 줄일 수 있지만 단일 seed의 날카로운 validation 성능을 희석할 수 있다.
3. latest main split에서는 LightGBM이 CatBoost보다 약간 좋았다.
4. rolling에서는 inner-selected CatBoost가 fold 1, 3에서 악화됐다.
5. weighted 후보는 main validation은 약하지만 rolling outer에서는 상대적으로 안정적이다.

현재 판단:

- main validation 최저 후보는 계속 기록한다.
- 이전 best CatBoost `5.658024`도 별도 best checkpoint로 유지한다.
- public 제출 횟수가 제한적이면 latest LightGBM `5.713983`보다 이전 CatBoost best를 우선한다.
- weighted 제출은 안정성 비교 후보로 유지한다.

---

## 16. 현재 코드의 전처리와 누수 방지

### 16.1 공통 전처리

tree residual pipeline:

1. timestamp 정렬
2. 중복 제거
3. hourly reindex
4. numeric column interpolation
5. forward fill
6. backward fill
7. train 기준 IQR threshold fit
8. feature clipping
9. train 기준 robust center / scale fit
10. robust z-score feature 생성

### 16.2 baseline과 residual target 분리

baseline:

```text
LOOKBACK = 168
midnight target
```

tree residual:

```text
ENSEMBLE_LOOKBACK = 672
ENSEMBLE_START_LOOKBACK = 672 + 168
all-hour train target
midnight validation/test target
```

### 16.3 누수 방지

- scaler와 clipping threshold는 train 구간에만 fit
- validation/test origin 이후 데이터는 feature로 사용하지 않음
- 과거 baseline error는 해당 96시간 실제값이 모두 관측된 경우만 사용
- 제출 target 이전 timestamp만 입력으로 사용

---

## 17. 자동 생성 로그와 결과 파일

로컬 실행 후 생성:

```text
submit_ensemble.csv
submit_anchor.csv
submit_lightgbm_residual.csv
submit_catboost_residual.csv
submit_weighted_anchor_lgbm_catboost.csv
ensemble_candidates.csv
ensemble_weights.csv
ensemble_lambda_search.csv
ensemble_backtest.csv
ensemble_horizon_summary.csv
ensemble_submission_distribution.csv
```

Git에는 결과 CSV를 커밋하지 않는다. 동일 환경에서 재생성할 수 있기 때문이다.

### 17.1 가장 먼저 확인할 로그

```text
ensemble_candidates.csv
ensemble_lambda_search.csv
ensemble_backtest.csv
```

### 17.2 추가로 확인할 로그

```text
ensemble_horizon_summary.csv
ensemble_submission_distribution.csv
```

---

## 18. 로컬 실행 방법

가상환경 활성화 없이 실행:

```powershell
cd C:\Users\sangw\2026_DeepSyS
.\.venv\Scripts\python.exe run_local_ensemble.py --threads 10
```

가상환경 활성화 후 실행:

```powershell
cd C:\Users\sangw\2026_DeepSyS
.\.venv\Scripts\Activate.ps1
python run_local_ensemble.py --threads 10
```

CatBoost GPU 시도:

```powershell
.\.venv\Scripts\python.exe run_local_ensemble.py --threads 10 --catboost-gpu
```

CatBoost 단일 seed 비교:

```powershell
.\.venv\Scripts\python.exe run_local_ensemble.py --threads 10 --catboost-seeds 42
```

빠른 smoke test:

```powershell
.\.venv\Scripts\python.exe run_local_ensemble.py --quick --threads 10
```

주의:

- `--quick`은 rolling backtest와 CatBoost를 생략한다.
- 최종 제출 후보를 만들 때는 `--quick`을 사용하지 않는다.

---

## 19. 실험 결과 요약

| 단계 | 핵심 변경 | validation MSE | rolling 결과 | public MSE | 판단 |
| --- | --- | ---: | --- | ---: | --- |
| seasonal baseline | threeway 24block shrink75 | `5.838997` | 안정적 | `7.17243` | 이후 anchor |
| Seq residual booster | GRU + attention, baseline residual | `5.645186` | mean improvement 음수 | - | 제출 제외 |
| PatchTST residual | custom patch encoder residual | `5.825482` | `-2.793064`, `0/3` | - | 실패 |
| tree residual v1 | midnight-only LightGBM/CatBoost | `5.562830` weighted | `-0.341884`, `0/3` | - | overfit |
| tree residual v2 | all-hour + error lag + horizon norm | `5.710577` CatBoost | weighted `+0.106609`, `3/3` | `6.62817` | 첫 유효 개선 |
| tree residual v3 | seed ensemble + lambda `~1.20` | `5.658024` CatBoost | 후보별 진단 필요 | 보고된 public `6.62817`과 연계 기록 | validation best |
| tree residual v4 | lambda `~1.60`, 후보별 제출 | `5.713983` LightGBM | selected `+0.024524`, `1/3`; weighted 상대 안정 | 미제출 | 이전 best보다 후퇴 |

주의:

- `tree residual v3` public score와 정확히 어떤 로컬 실행 artifact가 제출됐는지는 다음 세션에서 제출 파일 checksum 또는 별도 실험 ID로 관리하는 것이 좋다.
- 이후부터는 실행마다 experiment ID와 제출 파일명을 고정하는 것이 바람직하다.

---

## 20. 다음 개선 우선순위

### 20.1 실험 추적 강화

현재 가장 먼저 개선해야 할 것은 모델이 아니라 실험 추적이다.

추천:

- 실행 timestamp 또는 experiment ID를 파일명에 포함
- 후보 CSV에 다음 기록:
  - git commit
  - seeds
  - lambda grid
  - selected lambda
  - thread 수
  - CatBoost GPU 여부
- 제출 파일 checksum 기록
- Kaggle public score를 별도 markdown 또는 CSV에 연결

### 20.2 CatBoost seed 조합 비교

현재 3 seed ensemble이 항상 단일 seed보다 좋지 않다.

다음 비교:

```text
--catboost-seeds 42
--catboost-seeds 42,43,44
--catboost-seeds 42,2026,777
```

main validation과 rolling 후보별 MSE를 함께 본다.

### 20.3 lambda 범위 재검토

현재 LightGBM과 CatBoost 모두 `1.60` 상한에 걸릴 수 있다.

하지만 상한을 계속 확장하기 전에 다음을 확인해야 한다.

1. 단일 seed에서도 같은 경향인가.
2. rolling fold에서도 lambda가 커질수록 개선되는가.
3. submission distribution이 과격해지지 않는가.

validation 하나만 보고 무한정 lambda를 키우지 않는다.

### 20.4 weighted 후보 유지

weighted 후보는 main validation에서는 약하지만 rolling에서 안정적이다.

따라서 제출 파일을 유지한다.

```text
submit_weighted_anchor_lgbm_catboost.csv
```

### 20.5 딥러닝 모델 추가는 보류

N-HiTS, N-BEATS, Chronos-Bolt는 아직 구현하지 않았다.

이유:

- Seq residual과 PatchTST residual이 rolling에서 실패했다.
- tree residual이 더 저렴하고 실제 public 개선을 냈다.
- 현재는 tree residual 실험 추적과 seed/lambda 튜닝이 우선이다.

---

## 21. 새 세션에서 이어서 작업할 때 확인할 것

1. `MODEL_EXPERIMENT_HISTORY.md`를 읽는다.
2. `ensemble_candidates.csv`를 확인한다.
3. `ensemble_lambda_search.csv`를 확인한다.
4. `ensemble_backtest.csv`에서 후보별 outer MSE를 확인한다.
5. Kaggle에 실제 제출한 파일명을 확인한다.
6. public score를 해당 파일과 연결해 기록한다.
7. 이전 best와 비교한다.

현재 기록상 주요 best:

```text
seasonal public best = 7.17243
tree residual reported public best = 6.62817
validation best observed = 5.658024 (CatBoost residual, lambda=1.20)
latest validation selected = 5.713983 (LightGBM residual, lambda=1.60)
```

다음 목표:

```text
validation raw MSE <= 5.50
```

