# monitor_aware/ — bidders that are told how the monitors work

Produced by `addon_B/b2_aware_games.py` (`play`, then `score`). See `addon_B/README.md`.

Two conditions that are registered at run time in `addon_B/sdb_common.py`:

| Condition | What bidders are told |
|---|---|
| `dial_norule_aware` | group bonus (λ = 1), no rule stated, and how the three monitors work |
| `explicit_aware` | the benchmark's instructed ring, how the monitors work, and to stay below them |

## runs/

24 games, 12 rounds each, three agent families via OpenRouter (`or_deepseek`, `or_gptoss`, `or_llama`):
three seeds of each new condition per family, plus seeds 2 and 3 of the plain `explicit_collusion` ring so
that the instructed comparison has three games per family on each side. `usage.json` is the token usage
of the play step.

Each run folder has the usual `meta.json` and `rounds.jsonl`. `monitor/` holds the second monitoring
stack's files (`claims__j2_haiku.json`, `consistency.json`, `j2_haiku__saydo.json`) and the non-LLM
screens and merit tables.

Two games (`or_llama__explicit_aware__lam1__n3__s2` and `__s3`) each have one invalid round out of
twelve: the provider rate-limited one `act` call. The saved model panel has the same kind of loss
(4 of 1,080 rounds).

## analysis/

Output of `collusion_dial.evaluate` with the 48 benchmark games as `--logs` and, as `--dial-logs`, the
24 new games together with the 30 saved baseline games of the same three families (`dial`,
`dial_norule`, `dial_norule_watched`, `explicit_collusion` seed 1). Thresholds are frozen on the
benchmark. Within a folder, new games and baselines are scored by the same stack.

| Folder | Judge | Claim extractor |
|---|---|---|
| `analysis/j2_haiku__j2_haiku/` | Claude Haiku 5.5 | Claude Haiku 5.5 (the second stack of `results/second_stack/`) |
| `analysis/glm__qwen/` | GLM-5.3 | Qwen (the paper's own stack) |

For `glm__qwen` the baseline games use their saved scores and only the 24 new games were scored
(149 extractor calls, 24 judge calls, no failed call). The 30 baseline rows reproduce the cartel label
and the consistency, Say+Do and screen flags of `model_panel/analysis/panel_runs.csv` exactly.

## monitor_glm__qwen/

The paper's stack on the 24 new games, one folder per game: `claims__qwen.json`, `consistency.json`
(computed from those claims) and `glm__saydo.json`. They are kept apart from `runs/*/monitor/` because
`consistency.json` has one name per run folder and the one there belongs to the second stack. The
screens and merit tables do not depend on the stack.

## Per-condition tables

The per-condition table and the per-game rows of each stack are in
`addon_B/results_B/b2_aware_summary__<judge>__<extractor>.json` and
`b2_aware_per_game__<judge>__<extractor>.csv`; the printed output is in `b2_score_*_stdout.txt`.
