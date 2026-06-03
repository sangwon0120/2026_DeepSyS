# 2026 DeepSyS ETTh1 Forecasting

Electricity Transformer Temperature (ETT) 데이터셋의 `OT` 값을 예측하는 딥러닝 기반 시계열 예측 프로젝트입니다.

## Task

- 데이터: `ETTh1.csv`
- 입력 컬럼: `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`
- 예측 대상: 기준일 00시부터 미래 96시간의 `OT`
- 제출 형식: `sample_submit.csv`의 `ID`/`timestamp`별 `T0` ~ `T95`
- 평가 지표: MSE

## Leakage Rules

- `2018-02-01 00:00:00` 이후 데이터는 학습, 검증, 하이퍼파라미터 선택에 사용하지 않습니다.
- 각 제출 기준일 `D 00:00:00` 예측에는 `D 00:00:00` 이전 timestamp만 입력으로 사용합니다.
- scaling 통계는 train 구간에서만 계산합니다.
- validation과 backtest는 모두 target start가 00시인 window만 사용합니다.

## Current Approach

현재 메인 스크립트는 [ett_forecasting_pytorch.py](ett_forecasting_pytorch.py)입니다.

1. 규칙 기반 baseline 후보를 먼저 평가합니다.
   - `last_value`
   - `last_96h_repeat`
   - `last_week_same_time`
   - horizon block별 convex blend
2. 현재 강한 anchor baseline은 `threeway_last_prev_week_24block_shrink75`입니다.
3. 딥러닝 모델은 anchor baseline의 잔차를 학습합니다.
   - model: `ResidualMLP`
   - prediction: `baseline + lambda * residual_pred`
   - `lambda`는 validation MSE 기준으로 선택합니다.
4. ResidualMLP가 validation에서 baseline보다 좋을 때만 최종 predictor로 선택합니다.

## Known Scores

| Model / Predictor | Public MSE |
| --- | ---: |
| `last_value` | 8.33911 |
| `blend_last_value_last_week` | 7.30652 |
| `threeway_last_prev_week_4block` | 7.21015 |
| `threeway_last_prev_week_24block_shrink75` | 7.17243 |

현재 `ett_forecasting_pytorch.py`는 `ResidualMLP` residual 후보까지 포함합니다. ResidualMLP 제출 점수는 아직 별도로 기록하지 않았고, 위 public best는 제출 완료된 baseline anchor 기준입니다.

## Run

Colab 또는 Kaggle에서 다음처럼 실행합니다.

```python
import importlib
import ett_forecasting_pytorch as ett

ett = importlib.reload(ett)
result = ett.run(plot=True, residual_backtest=True)

print(result["predictor"]["name"])
print(result["metric"])
print(result["residual_lambda"])
print(result["residual_backtest"])
```

`residual_backtest=True`는 fold별 ResidualMLP 재학습을 수행하므로 시간이 더 걸립니다. 빠른 실행이 필요하면 `False`로 둘 수 있습니다.

기존 `SeqResidualBooster`를 로컬에서 학습하고 진단 CSV를 보관할 때는 wrapper를 사용합니다.

```bash
python run_local_pytorch.py
```

결과는 `csvFiles/pytorch_run_report.json`, `csvFiles/pytorch_*.csv`, `csvFiles/pytorch_experiments/<experiment-id>/`에 저장됩니다.

## Direct PatchTST Residual

직접 예측 실험 파이프라인은 [ett_multimodel_kaggle.py](ett_multimodel_kaggle.py)입니다.

- multivariate input + cyclic 시간 파생 변수
- 기본 anchor: `last_value`, 직전 `96h`, 전주·2주 전 동일 시각 blend를 섞은 leakage-safe three-way block blend
- anchor 1단계 분석: 전주·2주 전 blend, robust level-adjusted week, 제한된 block 복잡도 후보 비교
- 비교 옵션: `--anchor seasonal-naive`
- all-hour PatchTST residual pretrain
- midnight-origin PatchTST residual finetune
- PatchTST 수치 입력 window의 RevIN-style 정규화와 mean/std context feature
- validation 기준 residual lambda 탐색
- 시간순 validation segment `3/3` 안정성 gate와 `lambda=0` anchor fallback
- nested rolling backtest 기반 PatchTST residual 일반화 진단
- 선택 후보의 pre-test 전체 구간 final refit
- train/validation label purge, train-only scaler, test 입력 종료 시각 assert

이전 직접 멀티모델 비교는 `--legacy-multimodel` 옵션으로 실행할 수 있습니다.

로컬 실행:

```bash
.venv/bin/python ett_multimodel_kaggle.py
```

로컬 학습 smoke 실행:

```bash
.venv/bin/python ett_multimodel_kaggle.py --quick --output multimodel_output/submit_quick.csv --report-output multimodel_output/run_report_quick.json
```

PatchTST 학습 없이 anchor 1단계 후보만 비교:

```bash
python ett_multimodel_kaggle.py \
  --anchor-analysis-only \
  --output multimodel_output/submit_anchor_stage1_provisional.csv \
  --report-output multimodel_output/run_report_anchor_stage1.json
```

이 모드의 선택 결과는 provisional 후보입니다. rolling backtest를 통과하기 전에는 PatchTST 기본 anchor로 자동 승격하지 않습니다.

Anchor rolling backtest와 보수적 승격 gate 실행:

```bash
python ett_multimodel_kaggle.py \
  --anchor-rolling-backtest \
  --output multimodel_output/submit_anchor_stage2_rolling.csv \
  --report-output multimodel_output/run_report_anchor_stage2_rolling.json
```

현재 기본 anchor는 rolling `4/4` fold를 모두 개선한 `threeway_week2blend_12block_shrink50`입니다.

PatchTST residual nested rolling backtest 실행:

```bash
python ett_multimodel_kaggle.py \
  --patch-residual-rolling-backtest \
  --rolling-patch-seed 42 \
  --rolling-pretrain-epochs 3 \
  --rolling-finetune-epochs 8 \
  --rolling-patience 3 \
  --report-output multimodel_output/run_report_patch_residual_rolling_revin.json
```

PatchTST residual 모델은 기본적으로 입력 window 내부 수치 채널을 정규화하고, 각 window의 mean/std를 context feature로 함께 전달합니다. 시간에 따른 OT level shift 민감도를 줄이기 위한 옵션이며, 비교 실험에서는 `--no-patch-window-revin`으로 끌 수 있습니다.

PatchTST residual seed ensemble 실행:

```bash
.venv/bin/python ett_multimodel_kaggle.py --patch-seeds 42,2024,2026
```

`336`, `512` lookback과 평균 blend를 함께 비교:

```bash
.venv/bin/python ett_multimodel_kaggle.py --patch-seeds 42,2024,2026 --patch-lookbacks 336,512
```

학습 없이 누수 규칙과 제출 스키마를 빠르게 점검:

```bash
.venv/bin/python ett_multimodel_kaggle.py --validate-only --output multimodel_output/submit_validate_only.csv --report-output multimodel_output/run_report_validate_only.json
```

학습 실행 후 공유할 항목:

1. `multimodel_output/run_report.json`
2. Kaggle에 제출했다면 public MSE
3. 실행이 실패했다면 마지막 traceback

`multimodel_output/run_report.json`에는 실행 옵션, split 정보, seed별 epoch 이력, lambda 탐색, final refit 기록, 제출 파일 SHA-256이 포함됩니다.

로컬 `.venv`가 없다면 생성 후 필요한 패키지를 설치합니다.

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pandas torch lightgbm catboost matplotlib scikit-learn
```

## Outputs

- `csvFiles/sample_submit.csv`: 제출 형식 입력 파일
- `multimodel_output/submit.csv`: 직접 PatchTST residual 선택 후보
- `multimodel_output/submit_seasonal_naive.csv`: 비교용 24시간 반복 Seasonal Naive
- `multimodel_output/submit_anchor.csv`: 기본 strong three-way anchor
- `multimodel_output/submit_patchtst_residual.csv`: 직접 PatchTST residual 선택 후보 사본
- `multimodel_output/run_report.json`: 직접 예측 실행 설정, validation 결과, lambda 탐색, final refit 기록, 제출 checksum
- `csvFiles/ensemble_*.csv`: tree residual 진단 결과
- `csvFiles/experiments/<experiment-id>/`: tree residual 실행별 archive
- `best_residual_mlp.pt`: ResidualMLP checkpoint
- `best_model.pt`: legacy GRU checkpoint

직접 PatchTST 산출물은 `multimodel_output/`, 기존 tree residual 산출물은 `csvFiles/`에서 확인합니다. 재생성 가능하므로 git에는 포함하지 않습니다.
