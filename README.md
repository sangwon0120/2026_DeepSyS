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

## Outputs

- `submit.csv`: Kaggle 제출 파일
- `best_residual_mlp.pt`: ResidualMLP checkpoint
- `best_model.pt`: legacy GRU checkpoint

위 산출물은 재생성 가능하므로 git에는 포함하지 않습니다.
