# Control Framework

Hourly genetic-algorithm (GA) dispatch on a connected-tank network model,
plus minute-scale LightGBM level forecasting and a rule-based decision tree
for corrective feedback.

This package contains algorithms and illustrative configs only.
It does NOT include field measurements, trained model weights, hostnames,
credentials, database schemas, or any other operational infrastructure.

---

## Asset naming

| Role | ID |
|------|-----|
| Controlled pumps | `Pump-XL`, `Pump-HJT`, `Pump-XHJT`, `Pump-TJQ`, `Pump-WJT` |
| WWTPs | `WWTP-1`, `WWTP-2` |
| Upstream feeders | `Pump-U1` ... `Pump-U10` |
| Tank / pipe sections | `WJT-TJQ`, `TJQ-WWTP2`, `XHJT-WWTP2`, `HJT-XL`, `XL-WWTP1`, ... |

See `common/naming.py` for the full list.

---

## Layout

```text
control_framework/
|-- README.md
|-- requirements.txt
|-- common/                 Shared IDs, time helpers, offline data stubs
|-- 01_lgbm_level/          LightGBM train + recursive forecast
|-- 02_volume_analysis/     Level -> section volume / utilization
|-- 03_ga_dispatch/         Connected-tank model + GA hourly plan
|-- 04_feedback_control/    Limit scan + decision-tree actions
```

---

## Pipeline

```text
02 volume (V0) --> 03 GA + tank model --> hourly set-points
                                              |
01 LightGBM short-term level forecast --------+
                                              v
                                    04 decision-tree feedback
```

| Stage | Time scale | Method | Output |
|-------|------------|--------|--------|
| Volume analysis | event / minute | level-volume curve | section storage V0 |
| GA dispatch | hourly (24 h) | tank model + GA | discrete pump flows |
| Level forecast | 5 min x 12 | LightGBM level difference | short-term level path |
| Feedback | minute | rule decision tree | +/- unit recommendations |

---

## Quick start (offline)

```bash
pip install -r requirements.txt

# 1) Train demo LightGBM models (synthetic data unless offline_csv is set)
cd 01_lgbm_level
python train_lgbm.py
python level_forecast.py

# 2) Section volume demo
cd ../02_volume_analysis
python volume_analysis.py

# 3) GA dispatch demo
cd ../03_ga_dispatch
python main_dispatch.py

# 4) Feedback decision-tree demo
cd ../04_feedback_control
python main_scanning.py
```

To use your own (already anonymized) CSV for LightGBM, set `offline_csv` in
`01_lgbm_level/config.yml`. Expected columns follow `{StationID}_level` /
`{StationID}_flow`.

Trained weights are written locally to `01_lgbm_level/model/` and are NOT
shipped with this package.

---

## Module notes

### `01_lgbm_level`
- `train_lgbm.py` - offline training; saves `model/{StationID}.txt`
- `level_forecast.py` - recursive 5-min forecast using lagged flows + current level
- Feature relations are listed in `config.yml` (`data_relation`)

### `02_volume_analysis`
- Maps a station level to section volume via a monotonic curve (demo curve if no CSV)

### `03_ga_dispatch`
- `controloptimize.py`
  - `GeneticAlgorithm`: crossover / mutate / select
  - `PumpControl`: topology, WWTP-approach tank balancing, volume and storage penalties, switching penalty
- `main_dispatch.py` - offline GA loop with synthetic diurnal patterns

### `04_feedback_control`
- `decision_tree.py` - hand-crafted if/else tree (not sklearn)
- `main_scanning.py` - limit scan + strategy printout

---

## Security / disclosure policy

- Do NOT commit real SCADA exports, model weights trained on proprietary series,
  VPN endpoints, or database DSNs into this tree.
- Any production connector must live outside this repository and be injected
  at runtime by the operator.
- Config numeric limits here are illustrative placeholders, not site set-points.

---

## License / authorship

Research code extracted for manuscript support. Operational deployments remain
with the original project owners and are out of scope for this package.
