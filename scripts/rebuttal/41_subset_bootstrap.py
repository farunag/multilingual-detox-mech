#!/usr/bin/env python3
"""Paired bootstrap of steering effects split by data source
(original_human_parallel vs afrihate_human_reviewed) within xh and yo."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

TEST = Path("results/training_v2_400plus/seed_2024/test.csv")
METRICS = ["chrf_to_reference", "labse_to_reference", "labse_to_input", "copy_input"]
COMPS = [("english_transfer", "baseline"), ("within_language", "baseline"),
         ("english_transfer", "orthogonal_random_direction"),
         ("english_transfer", "negative_english_transfer"),
         ("english_transfer", "random_direction")]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=10000); a = ap.parse_args()
    run = Path(a.run_dir)
    src = pd.read_csv(TEST, keep_default_na=False).reset_index().rename(columns={"index": "row_index"})[["row_index", "source"]]
    m = pd.concat([pd.read_csv(p, keep_default_na=False) for p in sorted((run / "test").glob("*_metrics.csv"))])
    m = m.merge(src, on="row_index", how="left")
    assert m.source.notna().all()
    rng = np.random.default_rng(42); rows = []
    for lang in ["xh", "yo", "pooled"]:
        for s in ["original_human_parallel", "afrihate_human_reviewed", "all"]:
            d = m if lang == "pooled" else m[m.language == lang]
            d = d if s == "all" else d[d.source == s]
            for l, r in COMPS:
                if not {l, r} <= set(d.condition): continue
                L = d[d.condition == l].set_index("example_id").sort_index()
                R = d[d.condition == r].set_index("example_id").sort_index()
                assert list(L.index) == list(R.index)
                for met in METRICS:
                    diff = L[met].astype(float).to_numpy() - R[met].astype(float).to_numpy()
                    bs = diff[rng.integers(0, len(diff), (a.n_boot, len(diff)))].mean(1)
                    lo, hi = np.quantile(bs, [.025, .975])
                    rows.append(dict(language=lang, source=s, comparison=f"{l} - {r}", metric=met,
                                     n=len(diff), mean=diff.mean(), ci_low=lo, ci_high=hi,
                                     excludes_zero=bool(lo > 0 or hi < 0)))
    out = pd.DataFrame(rows); (run / "metrics").mkdir(exist_ok=True)
    out.to_csv(run / "metrics" / "paired_bootstrap_by_source.csv", index=False)
    print(out[(out.metric == "chrf_to_reference") & out.comparison.isin(["english_transfer - baseline", "english_transfer - orthogonal_random_direction"])].round(4).to_string(index=False))

if __name__ == "__main__":
    main()
