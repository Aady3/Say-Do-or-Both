"""Check a model endpoint before spending on it.

  python b_check_endpoint.py --key j2_haiku                       # one small JSON call
  python b_check_endpoint.py --equivalence --judge glm --extractor qwen
        Use after you point the keys `glm` or `qwen` at a different host (SWITCH A in models_extra.yaml).
        Six saved benchmark games, one per case type, are re-scored from scratch on the new host and
        compared with the saved scores. If scores move enough to flip a flag, the new host is NOT the
        same monitor: use it as a separate stack through b1_second_stack.py under a different key.
"""
import argparse
import asyncio
import json
import time

import pandas as pd

from sdb_common import BENCH_SOURCES, REPO, WORK, build_models, copy_runs, merged_config, run_dirs, run_monitor

ap = argparse.ArgumentParser()
ap.add_argument("--key")
ap.add_argument("--equivalence", action="store_true")
ap.add_argument("--judge", default="glm")
ap.add_argument("--extractor", default="qwen")
a = ap.parse_args()
cfg = merged_config()

if a.key:
    models, tracker = build_models(cfg, [a.key], 1)
    if models[a.key].is_mock:
        raise SystemExit("this key is the offline mock; there is no endpoint to check")
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}, "word": {"type": "string"}},
              "required": ["ok", "word"], "additionalProperties": False}
    t0 = time.time()
    res = asyncio.run(models[a.key].complete(
        [{"role": "system", "content": "You answer with JSON only."},
         {"role": "user", "content": 'Reply with {"ok": true, "word": "ready"}.'}], "check", schema, max_tokens=60))
    print(json.dumps({"data": res["data"], "error": res["error"], "usage": res["usage"],
                      "seconds": round(time.time() - t0, 1)}, indent=1))
    print("endpoint works" if res["data"] else "endpoint FAILED: see the error above")

if a.equivalence:
    thr = json.loads((REPO / "results/extended/analysis/eval_pooled/thresholds.json").read_text())
    dst = WORK / "equivalence" / f"{a.judge}__{a.extractor}"
    seen = set()

    def one_per_case(meta):
        if meta["seed"] == 1 and meta["condition"] not in seen:
            seen.add(meta["condition"])
            return True
        return False

    copy_runs(BENCH_SOURCES, dst, keep=one_per_case)
    saved = {}
    for d in run_dirs(dst):
        m = d / "monitor"
        saved[d.name] = {"consistency": json.loads((m / "consistency.json").read_text())["score"],
                         "saydo": max(json.loads((m / f"{a.judge}__saydo.json").read_text())["scores"].values())}
        for f in (m / f"claims__{a.extractor}.json", m / f"{a.judge}__saydo.json", m / "consistency.json"):
            f.unlink(missing_ok=True)
    run_monitor(dst, cfg, a.judge, a.extractor, ["saydo"], 3)
    rows = []
    for d in run_dirs(dst):
        m = d / "monitor"
        new_c = json.loads((m / "consistency.json").read_text())["score"]
        new_s = max(json.loads((m / f"{a.judge}__saydo.json").read_text())["scores"].values())
        rows.append({"game": d.name, "consistency_saved": saved[d.name]["consistency"], "consistency_new": new_c,
                     "flag_flips": (saved[d.name]["consistency"] >= thr["consistency"]) != (new_c >= thr["consistency"]),
                     "saydo_saved": saved[d.name]["saydo"], "saydo_new": new_s,
                     "saydo_flag_flips": (saved[d.name]["saydo"] >= thr["saydo"]) != (new_s >= thr["saydo"])})
    df = pd.DataFrame(rows)
    print(df.round(2).to_string(index=False))
    flips = int(df.flag_flips.sum() + df.saydo_flag_flips.sum())
    print(f"\nflags that changed: {flips} of {2 * len(df)}. "
          + ("Treat this host as the paper's monitor." if flips == 0 else
             "Do NOT treat this host as the paper's monitor; run it as a separate stack (b1_second_stack.py)."))
    print("(The judge samples at temperature 0.7, so small score differences are expected even on the same host.)")
