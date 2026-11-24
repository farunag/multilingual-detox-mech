#!/usr/bin/env python3
"""Rebuttal probe controls: pair-grouped, leave-one-language-out, within-target,
data-source subsets, and a dense layer sweep. Also reproduces the paper's
original random-split numbers at the paper's layers as a sanity check.

Reads the same test partition and prompt as scripts 23/31. One forward pass per
text stores the final-token state of every layer (hidden_states[k], i.e. the
input to block k), so all analyses share one extraction.
"""
from __future__ import annotations
import argparse, json, unicodedata
from pathlib import Path
import numpy as np, pandas as pd, torch
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODELS = {
    "aya": ("CohereLabs/aya-23-8B", Path("results/training_aya_23_8b")),
    "qwen": ("Qwen/Qwen2.5-7B-Instruct", Path("results/training_qwen2_5_7b")),
}
DATA = Path("results/training_v2_400plus/seed_2024/test.csv")
HR = ["am", "ar", "de", "en", "es", "hi", "uk"]
TGT = ["xh", "yo"]
SYSTEM_PROMPT = ("Rewrite toxic text to be non-toxic while preserving its meaning "
                 "and the original language. Return only the rewritten text.")

def norm(s): return " ".join(unicodedata.normalize("NFC", str(s)).split()).casefold()

def lr(): return LogisticRegression(max_iter=1000, random_state=42)

def auc_fit(Xtr, ytr, Xte, yte):
    if len(set(ytr)) < 2 or len(set(yte)) < 2: return np.nan
    p = lr().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, p))

def extract(model_key, seed, out_dir):
    cache = out_dir / f"states_{model_key}_seed{seed}.npz"
    df = pd.read_csv(DATA, keep_default_na=False)
    if cache.exists():
        z = np.load(cache); return df, z["tox"], z["det"]
    mid, root = MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(mid, use_fast=True)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(mid, quantization_config=q, torch_dtype=torch.bfloat16,
                                                device_map="auto", low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, root / f"seed_{seed}" / "adapter").eval()
    def states(text):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=512).to(model.device)
        with torch.inference_mode():
            hs = model(**ids, output_hidden_states=True).hidden_states
        return torch.stack([h[0, -1, :] for h in hs]).float().cpu().numpy().astype(np.float16)
    tox, det = [], []
    for i, r in df.iterrows():
        tox.append(states(str(r.toxic_input))); det.append(states(str(r.detox_output)))
        if (i + 1) % 100 == 0: print(f"  {i+1}/{len(df)}", flush=True)
    tox, det = np.stack(tox), np.stack(det)   # [n_pairs, n_layers+1, d]
    np.savez_compressed(cache, tox=tox, det=det)
    del model, base; torch.cuda.empty_cache()
    return df, tox, det

def analyses(df, tox, det, layers, model_key, seed):
    n = len(df)
    lang = df.language.to_numpy(); src = df.source.to_numpy()
    grp_pair = np.array([norm(t) for t in df.toxic_input])          # same toxic input -> same group
    L = np.concatenate([lang, lang]); S = np.concatenate([src, src]); G = np.concatenate([grp_pair, grp_pair])
    y = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    rows = []
    def add(layer, analysis, subset, auc, ntest, **kw):
        rows.append(dict(model=model_key, seed=seed, layer=layer, analysis=analysis,
                         subset=subset, auc=auc, n_test=ntest, **kw))
    for k in layers:
        X = np.concatenate([tox[:, k, :], det[:, k, :]]).astype(np.float32)
        # 1. original protocol (reproduces paper numbers)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        add(k, "pooled_random_original", "all", auc_fit(Xtr, ytr, Xte, yte), len(yte))
        hr = np.isin(L, HR); tg = np.isin(L, TGT)
        Xtr, Xte, ytr, yte = train_test_split(X[hr], y[hr], test_size=0.3, random_state=42, stratify=y[hr])
        add(k, "hr_random_original", "hr", auc_fit(Xtr, ytr, Xte, yte), len(yte))
        # 2. pair-grouped pooled and HR-only (5 random group splits)
        for name, mask in [("pooled_grouped", np.ones_like(hr)), ("hr_grouped", hr)]:
            gss = GroupShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
            Xm, ym, Gm = X[mask], y[mask], G[mask]
            for rep, (tr, te) in enumerate(gss.split(Xm, ym, Gm)):
                add(k, name, "all" if name.startswith("pooled") else "hr",
                    auc_fit(Xm[tr], ym[tr], Xm[te], ym[te]), len(te), rep=rep)
        # 3. transfer HR -> xh/yo (pair-disjoint by construction), overall and by language/source
        probe = lr().fit(X[hr], y[hr]); p = probe.predict_proba(X)[:, 1]
        add(k, "transfer_hr_to_target", "xh+yo", float(roc_auc_score(y[tg], p[tg])), int(tg.sum()))
        for lg in TGT:
            for s in ["all", "original_human_parallel", "afrihate_human_reviewed"]:
                m = (L == lg) & ((S == s) if s != "all" else True)
                if m.sum() and len(set(y[m])) == 2:
                    add(k, "transfer_hr_to_target", f"{lg}:{s}", float(roc_auc_score(y[m], p[m])), int(m.sum()))
        for s in ["original_human_parallel", "afrihate_human_reviewed"]:
            m = tg & (S == s)
            add(k, "transfer_hr_to_target", f"xh+yo:{s}", float(roc_auc_score(y[m], p[m])), int(m.sum()))
        # 4. leave-one-language-out over all nine languages
        for lg in HR + TGT:
            te = L == lg; tr = ~te
            add(k, "lolo", lg, auc_fit(X[tr], y[tr], X[te], y[te]), int(te.sum()))
        # 5. within-target ceiling: grouped 5-fold inside xh, yo, and xh+yo
        for name, m in [("xh", L == "xh"), ("yo", L == "yo"), ("xh+yo", tg)]:
            Xm, ym, Gm = X[m], y[m], G[m]
            for rep, (tr, te) in enumerate(GroupKFold(n_splits=5).split(Xm, ym, Gm)):
                add(k, "within_target_grouped", name, auc_fit(Xm[tr], ym[tr], Xm[te], ym[te]), len(te), rep=rep)
        print(f"  layer {k} done", flush=True)
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="dense", help="'dense' or comma list")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df, tox, det = extract(a.model, a.seed, out)
    nl = tox.shape[1]
    if a.layers == "dense":
        layers = sorted(set(range(2, nl, 2)) | {10, 15, 16, 20} & set(range(nl)))
    else:
        layers = [int(x) for x in a.layers.split(",")]
    res = analyses(df, tox, det, layers, a.model, a.seed)
    res.to_csv(out / f"probe_controls_{a.model}_seed{a.seed}.csv", index=False)
    summ = res.groupby(["layer", "analysis", "subset"], as_index=False).agg(
        auc_mean=("auc", "mean"), auc_sd=("auc", "std"), n_test=("n_test", "first"))
    summ.to_csv(out / f"probe_controls_{a.model}_seed{a.seed}_summary.csv", index=False)
    print(summ[summ.layer == 20].to_string(index=False))

if __name__ == "__main__":
    main()
