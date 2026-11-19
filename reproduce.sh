#!/usr/bin/env bash
# Reproduce all results of the paper. Run from the repository root after placing the raw data
# as described in data/README.md. Each stage can also be run on its own: ./reproduce.sh <stage>
#   stages: data | pretrain | hazards | figures | task1 | task2 | task3 | all (default)
# Large inputs stored elsewhere: LAYER_DIR=/path/to/layers HRRR_DIR=/path/to/hrrr ./reproduce.sh
set -euo pipefail
STAGE="${1:-all}"
LAYER_DIR="${LAYER_DIR:-data/raw/layers}"   # feature layers (override if stored elsewhere)
HRRR_DIR="${HRRR_DIR:-data/raw/hrrr}"       # HRRR GRIB2 folders
run() { echo; echo ">>> $*"; "$@"; }

if [[ "$STAGE" == data || "$STAGE" == all ]]; then
  for region in houston_harris texas; do
    run python scripts/data/build_dataset.py --region "$region" --layer-dir "$LAYER_DIR"   # Section 3.2: integration
    run python src/preprocessing.py --region "$region"            # Section 3.2: preprocessing
  done
fi

if [[ "$STAGE" == pretrain || "$STAGE" == all ]]; then
  run python src/model.py --region houston_harris   # H3-S2Vec (Harris)  -> embedding/Harris_Foundation_Model.parquet
  run python src/model.py --region texas            # H3-S2Vec (Texas)   -> embedding/Texas_Model.parquet
  run python src/model_centered.py                  # H3-S2Vec-C         -> embedding/Harris_Foundation_Model_C.parquet
fi

if [[ "$STAGE" == hazards || "$STAGE" == all ]]; then
  run python scripts/hazards/pde_labels.py          # Task 2 target
  run python scripts/hazards/mrms_rainfall.py       # Task 2 rainfall
  run python scripts/hazards/outage_labels.py       # Task 3 targets
  run python scripts/hazards/hrrr_extract.py --hrrr-dir "$HRRR_DIR"   # Task 3 wind, step 1
  run python scripts/hazards/hrrr_wind_features.py  # Task 3 wind, step 2
fi

if [[ "$STAGE" == figures || "$STAGE" == all ]]; then
  run python scripts/figures/make_figures.py        # Figures 1-6
fi

if [[ "$STAGE" == task1 || "$STAGE" == all ]]; then
  # Tables 2-3: Harris County, three embeddings
  for emb in Harris_Foundation_Model Texas_Model Harris_Foundation_Model_C; do
    run python downstream_task1.py --emb "embedding/$emb.parquet" \
        --dataset data/houston_harris_dataset.parquet --out "results/task1_harris_$emb.csv"
  done
  # Table 4: Texas-wide prediction with Texas-pretrained embeddings
  run python downstream_task1.py --emb embedding/Texas_Model.parquet \
      --dataset data/texas_dataset.parquet --out results/task1_texas_Texas_Model.csv
fi

if [[ "$STAGE" == task2 || "$STAGE" == all ]]; then
  # Table 6: experiment 1 (H3-S2Vec) and experiment 2 (H3-S2Vec-C)
  run python downstream_damage.py --emb embedding/Harris_Foundation_Model.parquet   --out results/task2_H3-S2Vec.csv
  run python downstream_damage.py --emb embedding/Harris_Foundation_Model_C.parquet --out results/task2_H3-S2Vec-C.csv
fi

if [[ "$STAGE" == task3 || "$STAGE" == all ]]; then
  # Tables 7-8: rows with Split = random in results/outage_cell_level.csv
  run python downstream_outage.py
fi
