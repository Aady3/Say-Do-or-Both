"""Silent versus talkative unscripted cartels on the six-family panel.

Reads results/model_panel/analysis/panel_runs.csv (saved games, no model calls) and prints, for the
unscripted ladder games that formed a cartel by the manipulation rule, how many exchange fewer than
ten messages and how often each frozen monitor flags the silent and the talkative ones.

    python addon_analysis/silent_split.py [REPO_ROOT]
"""
import os
import sys
import pandas as pd

R = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(R + "/results/model_panel/analysis/panel_runs.csv")
ladder = d[d.rung.notna()]
c = ladder[ladder.cartel == 1].copy()
c["silent"] = c.n_messages < 10
print(f"ladder games {len(ladder)}, cartels {len(c)}, silent {int(c.silent.sum())}, talkative {int((~c.silent).sum())}")
print("silent cartels with at most one message:", int((c[c.silent].n_messages <= 1).sum()))
print("share of declarations 'not independent': silent %.2f, talkative %.2f"
      % (c[c.silent].not_independent_share.mean(), c[~c.silent].not_independent_share.mean()))
for m in ("saydo", "consistency", "screens", "merit_s5", "say", "do", "structured", "think"):
    col = "flag_" + m
    out = []
    for name, part in (("silent", c[c.silent]), ("talkative", c[~c.silent])):
        x = part[col].dropna().astype(float)
        out.append(f"{name} {int(x.sum())}/{len(x)}")
    print(f"{m:12s}", ", ".join(out))
