# Rebuttal experiments (ARR #4657), 2026-11-24

Hardware: 1x RTX 3090 (Vast.ai). Transformers 4.57.6.
Runner: runner/run_rebuttal_vast.sh (python3 version as run). Full log: logs/run.log.

## Where the outputs are
- Probe controls (Aya-23-8B, random seeds 42 and 1337, layers 2-32): probe_controls/
  - analyses: pooled_random_original (paper protocol), pooled_grouped / hr_grouped (pair-grouped, 5 splits),
    transfer_hr_to_target (overall and by language x source), lolo (leave-one-language-out, 9 languages),
    within_target_grouped (grouped 5-fold inside xh, yo, xh+yo)
  - script: scripts/rebuttal/40_probe_controls.py
- Confirmatory steering replication, adapter random seed 1337:
  artifacts/confirmatory_aya_2026_09_30/replicate_seed1337/ (dev sweep, locked config, frozen test, 7 conditions, bootstrap; direction .pt files not kept, regenerate with script 34)
  - scripts: 34_aya_confirmatory_steering.py, scripts/rebuttal/35p_control_extension.py, 36p_paired_bootstrap.py
- Exploratory wider alpha {3,4,8}, dev only, adapter random seed 42:
  artifacts/confirmatory_aya_2026_09_30/seed42_wide_alpha_dev/
- Source-subset bootstrap (original human-written vs AfriHate-derived):
  .../primary_seed42/metrics/paired_bootstrap_by_source.csv and .../replicate_seed1337/metrics/paired_bootstrap_by_source.csv
  - script: scripts/rebuttal/41_subset_bootstrap.py
- Cached last-token hidden states (553 test pairs x 33 layers x 4096, float16, ~230 MB each) are NOT in git:
  see HF_DATASET.txt. 40_probe_controls.py reuses them if placed in the --out folder.

## Key numbers (layer 20; random seed 42 / 1337)
- Pooled AUC: random split 0.837 / 0.839; pair-grouped 0.879 / 0.882. Pair leakage did not inflate AUC.
- Transfer HR -> xh+yo: 0.595 / 0.597.
- LOLO: en .97/.96, hi .92/.91, uk .92/.92, de .90/.89, es .89/.89, ar .83/.81, am .59/.58, xh .61/.58, yo .56/.56.
  The three failing languages are the three outside Aya-23's 23 covered languages.
- Within-target grouped: xh .72/.77, yo .96/.98.
- Replication (seed 1337): dev selected k=20, alpha=0.25; all English-vector test comparisons null
  (pooled chrF vs baseline +0.0015 [-0.0018, 0.0055]). shuffled_pair == english_transfer (degenerate control; ignore).
- Seed-42 Yoruba chrF drop is confined to AfriHate-derived items: -0.046 [-0.076, -0.019] (n=41) vs +0.0005 (n=24).
- Wider alpha: monotone degradation; selection rule picks alpha=0.
- Environment drift: 9% (xh) / 13% (yo) of unsteered dev outputs differ from the September run (same adapter, greedy).
