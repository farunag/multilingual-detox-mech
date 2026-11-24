#!/usr/bin/env bash
# Rebuttal experiments for ARR #4657 on a single-GPU Vast instance.
# Usage (inside tmux):  HF_TOKEN=hf_xxx bash run_rebuttal_vast.sh
# Re-running skips steps that already finished.
set -euo pipefail
WORK=/workspace/detox_rebuttal
REPO=$WORK/repo
SCRIPTS_SRC=/workspace            # put the four .py files here
mkdir -p $WORK && cd $WORK
[ -d $REPO ] || git clone --depth 1 --branch confirmatory-aya-steering-2026-09-30 \
    https://github.com/farunawebservices/multilingual-detox-mech $REPO
cd $REPO
mkdir -p scripts/rebuttal && cp $SCRIPTS_SRC/{40_probe_controls,41_subset_bootstrap,35p_control_extension,36p_paired_bootstrap}.py scripts/rebuttal/
nvidia-smi --query-gpu=name,memory.total --format=csv
ls -la results/training_aya_23_8b/seed_{42,1337}/adapter/adapter_model.safetensors

python3 -m pip install -q -U "transformers>=4.44,<5" "peft>=0.12" "bitsandbytes>=0.43" accelerate sentence-transformers sacrebleu scikit-learn pandas
python3 -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])"

C=artifacts/confirmatory_aya_2026_09_30
OUT=$WORK/results/probe_controls
log(){ echo; echo "=== $(date -u +%H:%M) $* ==="; }

# Step A: probe controls, Aya random seeds 42 and 1337
for s in 42 1337; do
  [ -f $OUT/probe_controls_aya_seed${s}_summary.csv ] || { log "probes seed $s"; python3 scripts/rebuttal/40_probe_controls.py --model aya --seed $s --out $OUT; }
done

# Step B: replication with adapter random seed 1337, dev selection (same grid as paper)
RUN=replicate_seed1337; RD=$C/$RUN
[ -f $RD/config/dev_selection.json ] || { log "dev selection seed 1337"; python3 scripts/34_aya_confirmatory_steering.py --seed 1337 --phase dev --layers 16,20 --strengths 0.0,0.25,0.5,1.0,2.0 --run-name $RUN; }

# Step C: lock dev selection (written from dev results only), frozen test, extra controls, bootstrap
[ -f $RD/config/locked_test_configuration.json ] || python3 - <<PY
import json
sel = json.load(open("$RD/config/dev_selection.json"))["selected"]
lock = {"model_id": "CohereLabs/aya-23-8B", "adapter_seed": 1337,
        "direction_source": "English training rows only", "target_languages": ["xh", "yo"],
        "direction_estimation_split": "train.csv", "selection_split": "dev.csv", "evaluation_split": "test.csv",
        "selected_layer": int(sel["layer"]), "selected_alpha": float(sel["alpha"]),
        "selection_method": "development-only automatic selection (same rule as primary_seed42)",
        "frozen_test_not_used_for_selection": True}
json.dump(lock, open("$RD/config/locked_test_configuration.json", "w"), indent=2)
print(json.dumps(lock, indent=2))
PY
[ -f $RD/metrics/test_metrics_summary.csv ] || { log "frozen test seed 1337"; python3 scripts/34_aya_confirmatory_steering.py --seed 1337 --phase test --layers 16,20 --strengths 0.0,0.25,0.5,1.0,2.0 --run-name $RUN; }
[ -f $RD/test/orthogonal_random_direction_metrics.csv ] || { log "extra controls seed 1337"; RUN_DIR=$RD ADAPTER_SEED=1337 python3 scripts/rebuttal/35p_control_extension.py; }
RUN_DIR=$RD python3 scripts/rebuttal/36p_paired_bootstrap.py

# Step D: source-subset bootstrap, both runs (CPU)
for r in primary_seed42 replicate_seed1337; do python3 scripts/rebuttal/41_subset_bootstrap.py --run-dir $C/$r; done

# Step E: pack results (no weights) -> upload this zip to Claude
pack=$WORK/rebuttal_results; mkdir -p $pack
cp -r $RD $pack/ ; cp -r $C/primary_seed42/metrics $pack/primary_seed42_metrics
rsync -a --exclude '*.npz' $OUT/ $pack/probe_controls/
find $pack -name '*.pt' -delete
cd $WORK && rm -f rebuttal_results.zip && python3 -c "import shutil; shutil.make_archive('rebuttal_results','zip','rebuttal_results')"
log "DONE core set -> $WORK/rebuttal_results.zip"

# Step F (exploratory, runs last): wider alpha grid on DEV only, adapter random seed 42
cd $REPO
[ -f $C/seed42_wide_alpha_dev/dev/dev_summary.csv ] || { log "wide alpha dev"; python3 scripts/34_aya_confirmatory_steering.py --seed 42 --phase dev --layers 16,20 --strengths 0.0,3.0,4.0,8.0 --run-name seed42_wide_alpha_dev; }
cp -r $C/seed42_wide_alpha_dev $WORK/rebuttal_results/ && find $WORK/rebuttal_results -name '*.pt' -delete
cd $WORK && rm -f rebuttal_results.zip && python3 -c "import shutil; shutil.make_archive('rebuttal_results','zip','rebuttal_results')"
log "DONE all -> $WORK/rebuttal_results.zip"
