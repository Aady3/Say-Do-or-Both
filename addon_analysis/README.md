# addon_analysis: post-hoc analyses of the saved games

Four post-hoc analyses: the Holm correction and the cover-bid stress test behind the paper's Table 3
and Figure 3, the transfer table on the unscripted panel games, and the split of unscripted cartels into
silent and talkative ones. All scripts read the saved games and scores under `results/` and make no
model calls. Run them from the repository root:

    python addon_analysis/holm.py              # about 7 minutes; writes addon_analysis/holm.json
    python addon_analysis/stress.py            # writes addon_analysis/stress.csv
    python addon_analysis/panel_unscripted.py  # writes addon_analysis/panel_unscripted.json
    python addon_analysis/silent_split.py      # prints only

Each script also accepts the path to the repository root as an optional argument, so it can be run from
any folder. They need numpy, pandas, scikit-learn and matplotlib (the repository's own
`requirements.txt`). The outputs behind the paper are stored in this folder. `holm.py` is seeded and
reproduces `holm.json` exactly; for `stress.py` see the note on reproducibility below.

## holm.py: multiple-comparison correction of the seven delta tests

Rebuilds the paired bootstrap of delta = AUROC(monitor) - max(AUROC(Say), AUROC(Do)) on the 48 pooled
games (`results/extended/analysis/eval_pooled/`), with the same seed and 2,000 resamples as
`collusion_dial/evaluate.py`, for seven monitors: Say+Do, Structured, Consistency, Classifier (bids
only), Classifier, the version with statement-style features, and Think. It prints nominal one-sided
p-values, Holm-adjusted p-values and Bonferroni-adjusted intervals, and repeats everything with 20,000
resamples as a stability check. It first checks that it reproduces the saved `hybrid_do` scores.
Output: `holm.json` (keys `<resamples>_<monitor>`; fields `d`, `lo`, `hi` = delta and its 95% interval,
`p`, `holm`, `alo`, `ahi` = Bonferroni-adjusted interval).

Result (2,000 resamples): Think 0.0005 -> 0.0035; Classifier 0.0025 -> 0.015; Classifier, bids only
0.003 -> 0.015; statement-style version 0.0195 -> 0.078; Say+Do 0.33 -> 0.99; Structured and
Consistency -> 1.00. Three tests survive at 0.05.

Monitor names in the script and in the paper:

| Script | Paper | Features |
|---|---|---|
| `clf` | Classifier | `say`, `do` and `consistency` families |
| `clf_bid` | Classifier, bids only | `do` family (the repository's `hybrid_do`) |
| `hybrid` | + statement style | all four families, including `internal` (the repository's `hybrid`) |
| `saydo`, `structured`, `think`, `consistency` | Say+Do (raw), Structured report, Think (upper bound), Consistency | saved scores in `monitor_scores.csv` |

## stress.py: counterfactual cover-bid stress test

For every scripted ring, each losing bid of a sister firm is moved a fraction alpha of the way towards
the winning bid plus a margin drawn from the honest games' losing margins. Bid features are recomputed
for the same pair of firms; messages, consistency features and all innocent games are unchanged. The
script reports the AUROC of the highest-sister-bid baseline, the bid-only classifier and the full
classifier, both frozen on the original games and refitted on the perturbed games (leave-one-seed-out),
averaged over 20 random draws. It reads the 48 games from `results/main/runs/eval` and
`results/extended/runs/eval_b2`. Output: `stress.csv` (the data behind the stress-test figure).

It first checks that it reproduces the saved bid features exactly and the 0.96 one-feature baseline.

Reproducibility: the random draws are seeded, but the pool of honest losing margins is built in the
order in which the operating system lists the run folders, so the same seed picks different margins on
different machines. The alpha = 0 row is exact everywhere. For alpha > 0 a re-run moved the mean AUROCs
by at most about 0.01, inside the standard deviations in the `_sd` columns. The stored `stress.csv` is
the run reported in the paper; running the script overwrites it.

Columns of `stress.csv` (means over the draws; `_sd` columns are standard deviations):

| Column | Meaning |
|---|---|
| `alpha` | camouflage fraction: 0 = logged cover bids, 1 = distributed like honest losing bids |
| `base` | AUROC of the one-feature baseline, the highest bid a sister firm placed in the game |
| `bid_frozen`, `clf_frozen` | AUROC of the bid-only and the full classifier fitted on the original games |
| `bid_refit`, `clf_refit` | AUROC of the same two classifiers refitted on the perturbed games |
| `rec_clf_frozen` | recall of the frozen full classifier at a 0.5 threshold |
| `covmed` | median gap between a sister's losing bid and the winning bid in the scripted rings |

## panel_unscripted.py: transfer on the unscripted panel games only

    python addon_analysis/panel_unscripted.py    # writes addon_analysis/panel_unscripted.json

Reads the saved panel runs and scores (no model calls). It recomputes the pair features of the 90 panel
games from the saved rounds and cached claims, checks that this reproduces the saved frozen scores of
`hybrid` and `hybrid_do` exactly, and then scores the classifier without the statement-style family,
frozen on the 48 benchmark games.

It prints, and stores in `panel_unscripted.json`:

- the transfer table on all 90 panel games (unscripted games plus the 18 scripted controls);
- the transfer table on the 72 unscripted games only (42 with a cartel, 30 without), which is the
  table in the paper;
- the same table with cartels labelled by the collusion index alone. Nine of the 42 unscripted cartels
  carry the label only through the "at least two executed agreements" route, which shares its input
  with the consistency check; with the index-only label consistency falls from 0.83 to 0.72 AUROC and
  the classifier stays at 0.85;
- the highest-sister-bid baseline on the panel (0.73 on all 90 games, 0.71 on the 72 unscripted);
- flags on the 14 unscripted games without a cartel that have ten or more messages;
- benchmark facts quoted in the paper: all 4,536 agent calls parsed; share of rounds won by the sister
  firms and mean overcharge (winning bid minus second-lowest cost) per case type; cheap-talk scores and
  flags of Say and consistency at the leave-one-seed-out and at the frozen thresholds.

`panel_unscripted.log` is the printed output of the run behind the paper. A re-run reproduces
`panel_unscripted.json` exactly; in the log only the two "max abs diff" lines can differ, at the level
of floating-point round-off (1e-16).

## silent_split.py: silent versus talkative unscripted cartels

    python addon_analysis/silent_split.py

Reads `results/model_panel/analysis/panel_runs.csv` and splits the 42 unscripted panel cartels into
those with fewer than ten messages (6) and the rest (36), then prints how often each frozen monitor
flags each group. These are the numbers in the paper's paragraph on silent deference.
