# Leakage Note

- Train/validation/model-selection targets are restricted to windows ending before `2018-02-01 00:00:00`.
- The scaler mean/std are fit only on each training split, never on validation or test rows.
- Validation and rolling backtest targets always start at `00:00`.
- For each target start `D 00:00:00`, the input window ends at `D 00:00:00 - 1 hour`.
- Submission inference asserts that the latest input timestamp is strictly earlier than the row ID timestamp.
- Later submission rows may use earlier test-period observations only when those observations occur before the row's target timestamp, matching the project rule.
- Current script method: `threeway_last_prev_week_24block_shrink75` anchor plus optional `ResidualMLP` residual correction selected by validation MSE.
- Best recorded public MSE so far: `7.17243` from `threeway_last_prev_week_24block_shrink75`.
