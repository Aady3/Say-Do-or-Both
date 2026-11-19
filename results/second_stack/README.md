# second_stack/ — the 48 benchmark games re-scored by other judges

Produced by `addon_B/b1_second_stack.py`. No new games: the saved transcripts of
`main/runs/eval` and `extended/runs/eval_b2` are read by a different judge and a different claim
extractor, and `collusion_dial.evaluate` is rerun.

| Folder | Judge | Claim extractor |
|---|---|---|
| `analysis/j2_haiku__j2_haiku/` | Claude Haiku 5.5 | Claude Haiku 5.5 |
| `analysis/j2_gemini__j2_haiku/` | Gemini 3.1 Flash Lite | Claude Haiku 5.5 (the same extracted claims as the row above) |

Both judges read the `say`, `do`, `saydo` and `structured` views. Model entries are in
`addon_B/models_extra.yaml`. Each folder has the layout described in `results/README.md`
("An analysis folder"). The side-by-side tables against the paper's stack (GLM judge, Qwen extractor,
`extended/analysis/eval_pooled`) are in `addon_B/results_B/b1_second_stack__*.json`.

The per-game judge outputs are not versioned; rerunning the script regenerates them under
`addon_B/work/`.
