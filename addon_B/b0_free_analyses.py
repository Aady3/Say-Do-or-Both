"""B0. Three analyses on saved data. No model calls, no cost. Already run; outputs are in results_B/.

  python b0_free_analyses.py

1. Is the classifier better than the one-feature baseline (the highest bid a sister firm placed)?
   Paired, stratified bootstrap of AUROC(classifier) - AUROC(baseline) on the 48 benchmark games.
2. Does a simple fusion of the SEPARATE Say and Do judge scores beat the judge that reads both?
   mean, max, and a two-feature logistic regression fitted leave-one-seed-out, each against the best
   single channel with the same bootstrap as the paper's delta table.
3. What happens to cross-agent consistency under the cover-bid stress test?
   The scripted sisters' losing bids are moved towards an honest-looking margin (as in stress.py),
   the messages are left as they were, and the saved claims are re-scored against the new bids.
"""
import glob
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from sdb_common import OUT, REPO, save
from collusion_dial.metrics.collusion import load_run
from collusion_dial.monitors.claims import score_claims
from collusion_dial.monitors.features import FAMILIES
from collusion_dial.monitors.hybrid import _model, fit_predict_loso

A = REPO / "results/extended/analysis/eval_pooled"
ms = pd.read_csv(A / "monitor_scores.csv")
f = pd.read_csv(A / "features.csv")
thr = json.loads((A / "thresholds.json").read_text())
piv = ms[ms.stake.isna()].pivot_table(index="run_id", columns="monitor", values="score")
f = f.set_index("run_id").loc[piv.index].reset_index()
lab, seeds = f.label.to_numpy(int), f.seed.to_numpy()
pos, neg = np.where(lab == 1)[0], np.where(lab == 0)[0]
runs = {}
for mp in glob.glob(str(REPO / "results/main/runs/eval/*/meta.json")) + glob.glob(str(REPO / "results/extended/runs/eval_b2/*/meta.json")):
    d = mp[:-len("meta.json")]
    meta, rounds = load_run(d)
    runs[meta["run_id"]] = (meta, rounds, d)
assert set(f.run_id) <= set(runs)
res = {}


def maxsis(meta, rounds):
    return max(r["bids"][s] for r in rounds if r["valid"] for s in meta["sisters"] if r["bids"].get(s) is not None)


def boot(scores, n, seed):
    """Stratified resampling of games, the scheme of collusion_dial.evaluate and holm.py."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in scores}
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for k, v in scores.items():
            out[k].append(roc_auc_score(lab[idx], v[idx]))
    return {k: np.asarray(v) for k, v in out.items()}


def report(name, d):
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = float((d <= 0).mean())
    print(f"   {name:34s} CI95 [{lo:+.3f}, {hi:+.3f}]  one-sided p = {p:.4f}")
    return {"ci_low": float(lo), "ci_high": float(hi), "p_one_sided": p}


# ------------------------------------------------------------------ 1
NOINT = FAMILIES["say"] + FAMILIES["do"] + FAMILIES["consistency"]
clf, _ = fit_predict_loso(f, NOINT)
clf_bid, _ = fit_predict_loso(f, FAMILIES["do"])
base = np.array([maxsis(*runs[r][:2]) for r in f.run_id], float)
S = {"clf": clf, "clf_bid": clf_bid, "baseline": base}
print("1. Classifier against the one-feature baseline (highest sister bid)")
pt = {k: roc_auc_score(lab, v) for k, v in S.items()}
print("   AUROC:", {k: round(v, 3) for k, v in pt.items()})
res["baseline"] = {"auroc": pt}
for n, seed in ((2000, 0), (20000, 1)):
    b = boot(S, n, seed)
    print(f"   {n} resamples")
    res["baseline"][f"clf_minus_baseline_{n}"] = {"point": pt["clf"] - pt["baseline"], **report("classifier - baseline", b["clf"] - b["baseline"])}
    res["baseline"][f"clf_bid_minus_baseline_{n}"] = {"point": pt["clf_bid"] - pt["baseline"], **report("bids-only classifier - baseline", b["clf_bid"] - b["baseline"])}

# ------------------------------------------------------------------ 2
say, do, saydo = (piv[c].to_numpy(float) for c in ("say", "do", "saydo"))
X = np.c_[say, do]
lr = np.full(len(lab), np.nan)
for s in sorted(set(seeds)):
    tr, te = seeds != s, seeds == s
    lr[te] = _model().fit(X[tr], lab[tr]).predict_proba(X[te])[:, 1]
F = {"say": say, "do": do, "saydo_judge": saydo, "mean(say,do)": (say + do) / 2, "max(say,do)": np.maximum(say, do),
     "logistic(say,do), leave-one-seed-out": lr}
print("\n2. Fusion of the separate Say and Do judge scores")
ptf = {k: roc_auc_score(lab, v) for k, v in F.items()}
print("   AUROC:", {k: round(v, 3) for k, v in ptf.items()})
best = max(ptf["say"], ptf["do"])
res["fusion"] = {"auroc": ptf}
b = boot(F, 2000, 0)
bb = np.maximum(b["say"], b["do"])
for k in ("saydo_judge", "mean(say,do)", "max(say,do)", "logistic(say,do), leave-one-seed-out"):
    res["fusion"][k] = {"delta_vs_best_single": ptf[k] - best, **report(f"{k} - best single (delta {ptf[k] - best:+.3f})", b[k] - bb)}
res["fusion"]["logistic_minus_saydo_judge"] = {"point": ptf["logistic(say,do), leave-one-seed-out"] - ptf["saydo_judge"],
                                               **report("logistic fusion - Say+Do judge", b["logistic(say,do), leave-one-seed-out"] - b["saydo_judge"])}

# ------------------------------------------------------------------ 3
claims = {}
for rid, (meta, rounds, d) in runs.items():
    cp = sorted(glob.glob(d + "monitor/claims__qwen.json"))
    claims[rid] = json.load(open(cp[0]))["items"] if cp else None
marg = []
for rid, (meta, rounds, d) in runs.items():
    if meta["condition"] in ("honest", "honest_nomsg"):
        for r in rounds:
            if r["valid"] and r["winning_bid"] is not None:
                marg += [b_ - r["winning_bid"] for k, b_ in r["bids"].items() if b_ is not None and k != r["winner"]]
marg = np.array(marg)


def perturb(meta, rounds, alpha, rng):          # identical to addon_analysis/stress.py
    out = []
    for r in rounds:
        r = dict(r)
        r["bids"] = dict(r["bids"])
        if r["valid"] and r["winning_bid"] is not None:
            for s in meta["sisters"]:
                b_ = r["bids"].get(s)
                if b_ is None or s == r["winner"]:
                    continue
                tgt = r["winning_bid"] + max(0.5, float(rng.choice(marg)))
                if b_ > tgt:
                    r["bids"][s] = round(b_ - alpha * (b_ - tgt), 1)
        out.append(r)
    return out


def cons(rid, rounds):
    meta = runs[rid][0]
    c = claims[rid]
    if c is None:
        return 0.0, 0
    sc = score_claims(c, rounds, meta["firms"])
    ring = sc["pairs"].get("-".join(sorted(meta["sisters"])), {})
    return float(sc["score"]), int(ring.get("executed_agreements", 0))


saved = piv["consistency"].to_numpy(float)
orig = np.array([cons(r, runs[r][1])[0] for r in f.run_id])
print("\n3. Cross-agent consistency under the cover-bid stress test")
print("   max abs difference from the saved consistency scores at alpha = 0: %.2e" % np.abs(orig - saved).max())
explicit = np.array([runs[r][0]["condition"] == "explicit_collusion" for r in f.run_id])
rows = []
for alpha in (0, 0.25, 0.5, 0.75, 1.0):
    au, rec, ex = [], [], []
    for rs in range(20):
        rng = np.random.default_rng(100 + rs)
        sc, exe = [], []
        for rid, l in zip(f.run_id, lab):
            meta, rounds, _ = runs[rid]
            if l == 1 and alpha > 0:
                rounds = perturb(meta, rounds, alpha, rng)
            s_, e_ = cons(rid, rounds)
            sc.append(s_)
            exe.append(e_)
        sc, exe = np.array(sc), np.array(exe)
        au.append(roc_auc_score(lab, sc))
        rec.append(float((sc[explicit] >= thr["consistency"]).mean()))
        ex.append(float(exe[explicit].mean()))
        if alpha == 0:
            break
    rows.append({"alpha": alpha, "consistency_auroc": float(np.mean(au)),
                 "explicit_rings_flagged_at_frozen_threshold": float(np.mean(rec)),
                 "executed_agreements_per_explicit_ring": float(np.mean(ex))})
    print("  ", {k: round(v, 3) for k, v in rows[-1].items()})
pd.DataFrame(rows).to_csv(OUT / "b0_consistency_under_stress.csv", index=False)
res["consistency_under_stress"] = rows
save("b0_free_analyses.json", res)
