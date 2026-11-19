"""B1. Re-score the 48 benchmark games with a second judge and a second claim extractor.

No new games are played. The saved transcripts are copied to work/ and read by a different model.

  python b1_second_stack.py --dry-run                        # what would be called, rough cost
  python b1_second_stack.py                                  # judge j2_haiku + extractor j2_haiku
  python b1_second_stack.py --judge j2_gemini --extractor qwen   # second judge, the paper's saved claims
  python b1_second_stack.py --judge mock --extractor mock    # offline pipeline test, no credits

Why: the paper's headline negative result (a judge that reads messages and bids shows no detectable
gain over the best single channel) rests on one judge model and one prompt, and every claim-based
monitor rests on one extractor. This reruns both on a second family and reports the same table.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sdb_common import BENCH_SOURCES, OUT, REPO, WORK, copy_runs, merged_config, run_dirs, run_evaluate, run_monitor, save, usd
from collusion_dial.evaluate import youden_threshold
from collusion_dial.monitors.features import FAMILIES
from collusion_dial.monitors.hybrid import fit_predict_loso
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--judge", default="j2_haiku")
ap.add_argument("--extractor", default="j2_haiku", help="'qwen' reuses the paper's saved claims (no extraction calls)")
ap.add_argument("--views", default="say,do,saydo,structured")
ap.add_argument("--concurrency", type=int, default=6)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
views = [v for v in a.views.split(",") if v]
cfg = merged_config()
stack = WORK / f"stack_{a.judge}__{a.extractor}"
bench = stack / "bench"
n = copy_runs(BENCH_SOURCES, bench)
dirs = run_dirs(bench)
print(f"{n} benchmark games copied to {bench}")

if a.dry_run:
    need_j = sum(not (d / "monitor" / f"{a.judge}__{v}.json").exists() for d in dirs for v in views)
    need_e = sum(not (d / "monitor" / f"claims__{a.extractor}.json").exists() for d in dirs)
    tok = []
    for d in dirs:                                   # input size of the saved judge calls, as a guide
        for v in views:
            p = d / "monitor" / f"glm__{v}.json"
            if p.exists():
                tok.append(json.loads(p.read_text()).get("usage", {}).get("input_tokens", 0))
    mean_in = float(np.mean(tok)) if tok else 4000.0
    c = cfg["models"].get(a.judge, {})
    pin, pout = c.get("price_per_m_input"), c.get("price_per_m_output")
    est = (need_j * mean_in * pin + need_j * 500 * pout) / 1e6 if pin is not None and pout is not None else None
    print(f"judge calls still needed: {need_j} (mean input about {mean_in:.0f} tokens each)")
    print(f"games still needing claim extraction: {need_e} (a few calls per game, similar size)")
    print("rough judge cost: " + (f"${est:.2f}; extraction about the same again" if est is not None else "price unknown for this key"))
    raise SystemExit

usage = run_monitor(bench, cfg, a.judge, a.extractor, views, a.concurrency)
out = run_evaluate(bench, stack / "eval", a.judge, a.extractor)

# ------------------------------------------------------------------ summary
NOINT = FAMILIES["say"] + FAMILIES["do"] + FAMILIES["consistency"]
PAPER = REPO / "results/extended/analysis/eval_pooled"


def table(folder):
    mon = pd.read_csv(folder / "monitors.csv").set_index("monitor")
    feat = pd.read_csv(folder / "features.csv")
    comp = json.loads((folder / "channel_comparison.json").read_text())
    rows = {}
    for m in ("say", "do", "saydo", "structured", "consistency", "hybrid_do", "hybrid", "screens"):
        if m in mon.index:
            r = mon.loc[m]
            rows[m] = {"auroc": float(r.auroc), "recall": float(r.recall), "fpr": float(r.fpr)}
    sc, _ = fit_predict_loso(feat, NOINT)
    lab = feat.label.to_numpy(int)
    th = youden_threshold(feat.label, sc)
    rows["classifier"] = {"auroc": float(roc_auc_score(lab, sc)), "recall": float((sc[lab == 1] >= th).mean()),
                          "fpr": float((sc[lab == 0] >= th).mean())}
    best = max(rows["say"]["auroc"], rows["do"]["auroc"])
    rows["classifier"]["delta_vs_best_single"] = rows["classifier"]["auroc"] - best
    deltas = {k: comp[k] for k in ("delta_raw", "delta_struct", "delta_consistency", "delta_hybrid") if k in comp}
    return rows, deltas


new_rows, new_d = table(out)
old_rows, old_d = table(PAPER)
ms_new = pd.read_csv(out / "monitor_scores.csv")
ms_old = pd.read_csv(PAPER / "monitor_scores.csv")
print(f"\nSecond stack: judge = {a.judge}, extractor = {a.extractor}   (paper: judge glm, extractor qwen)")
print(f"{'monitor':14s} {'paper AUROC':>12s} {'stack 2 AUROC':>14s} {'paper rec/FPR':>15s} {'stack 2 rec/FPR':>16s}")
for m in ("say", "do", "saydo", "structured", "consistency", "hybrid_do", "classifier", "hybrid", "screens"):
    if m in new_rows and m in old_rows:
        o, n_ = old_rows[m], new_rows[m]
        print(f"{m:14s} {o['auroc']:12.3f} {n_['auroc']:14.3f} {o['recall']:9.2f}/{o['fpr']:.2f} {n_['recall']:11.2f}/{n_['fpr']:.2f}")
print("\nDelta against the best single channel (95% bootstrap interval, one-sided p):")
for k, lab_ in (("delta_raw", "Say+Do judge"), ("delta_struct", "Structured"), ("delta_consistency", "Consistency"), ("delta_hybrid", "Classifier + statement style")):
    for nm, d in (("paper  ", old_d), ("stack 2", new_d)):
        if k in d:
            x = d[k]
            print(f"  {lab_:30s} {nm}: {x['point']:+.3f} [{x['ci_low']:+.3f}, {x['ci_high']:+.3f}] p = {x['p_one_sided']:.4f}")
print(f"  {'Classifier (no statement style)':30s} stack 2: {new_rows['classifier']['delta_vs_best_single']:+.3f}  (paper: {old_rows['classifier']['delta_vs_best_single']:+.3f})")
# fusion of the SEPARATE Say and Do scores of this judge (see b0_free_analyses.py, part 2)
fus = {}
pv = ms_new[ms_new.stake.isna()].pivot_table(index="run_id", columns="monitor", values="score")
lab_s = pd.read_csv(out / "runs.csv").set_index("run_id").label.reindex(pv.index)
if {"say", "do", "saydo"} <= set(pv.columns):
    ok = pv[["say", "do", "saydo"]].notna().all(axis=1) & lab_s.notna()
    y = lab_s[ok].to_numpy(int)
    sc = {"say": pv.say[ok].to_numpy(float), "do": pv.do[ok].to_numpy(float), "saydo": pv.saydo[ok].to_numpy(float)}
    sc["mean"] = (sc["say"] + sc["do"]) / 2
    pos_, neg_ = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(0)
    d_mean, d_judge = [], []
    for _ in range(2000):
        i = np.concatenate([rng.choice(pos_, len(pos_)), rng.choice(neg_, len(neg_))])
        au = {k: roc_auc_score(y[i], v[i]) for k, v in sc.items()}
        d_mean.append(au["mean"] - max(au["say"], au["do"]))
        d_judge.append(au["mean"] - au["saydo"])
    pt_ = {k: roc_auc_score(y, v) for k, v in sc.items()}
    fus = {"auroc_mean_of_say_and_do": float(pt_["mean"]), "delta_vs_best_single": float(pt_["mean"] - max(pt_["say"], pt_["do"])),
           "ci": [float(x) for x in np.percentile(d_mean, [2.5, 97.5])], "p_one_sided": float((np.asarray(d_mean) <= 0).mean()),
           "mean_minus_saydo_judge": float(pt_["mean"] - pt_["saydo"]),
           "mean_minus_saydo_judge_ci": [float(x) for x in np.percentile(d_judge, [2.5, 97.5])], "n_games": int(ok.sum())}
    print(f"\nAverage of the separate Say and Do scores of this judge: AUROC {fus['auroc_mean_of_say_and_do']:.3f}, "
          f"delta vs best single {fus['delta_vs_best_single']:+.3f} [{fus['ci'][0]:+.3f}, {fus['ci'][1]:+.3f}] p = {fus['p_one_sided']:.4f} "
          f"(paper's judge: 0.975, +0.084 [+0.018, +0.138], p = 0.0035)")
agree = {}
for v in views + ["consistency"]:
    x = ms_old[(ms_old.monitor == v) & ms_old.stake.isna()].set_index("run_id").score
    y = ms_new[(ms_new.monitor == v) & ms_new.stake.isna()].set_index("run_id").score
    j = x.index.intersection(y.index)
    if len(j) > 5:
        agree[v] = {"spearman": float(spearmanr(x[j], y[j]).statistic), "n": int(len(j))}
print("\nAgreement of the two stacks on the same games (Spearman of scores):", {k: round(v["spearman"], 2) for k, v in agree.items()})
failed = ms_new[(~ms_new.ok.astype(bool)) & ms_new.monitor.isin(views)]
print(f"judge replies that failed to parse: {len(failed)} of {int(ms_new.monitor.isin(views).sum())}")
mat = pd.read_csv(out / "case_type_matrix.csv", index_col=0)
print("\nShare of games flagged per case type (stack 2):")
print(mat.loc[[m for m in ("say", "do", "saydo", "structured", "consistency", "hybrid") if m in mat.index]].round(2).to_string())
save(f"b1_second_stack__{a.judge}__{a.extractor}.json",
     {"judge": a.judge, "extractor": a.extractor, "stack2": new_rows, "paper": old_rows, "stack2_deltas": new_d,
      "paper_deltas": old_d, "fusion_mean_of_say_and_do": fus, "agreement": agree, "parse_failures": int(len(failed)),
      "case_type_matrix": mat.round(3).to_dict("index"), "usage": usage, "estimated_usd": usd(usage)})
