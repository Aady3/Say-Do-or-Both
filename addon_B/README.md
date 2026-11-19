# Add-on analyses, the second monitoring stack and monitor-aware bidders

None of these scripts edits a file of the repository: run folders are copied to `work/` first and outputs
go to `results_B/`. Run them from this folder, placed inside the repository root (or set
`SDB_REPO=/path/to/repository`), after loading the repository's `.env`.

| Script | Model calls | What it does |
|---|---|---|
| `b0_free_analyses.py` | none | (1) Paired bootstrap of the classifier against the one-feature baseline. (2) Fusion of the separate Say and Do judge scores (mean, max, two-feature logistic regression) against the best single channel and against the Say+Do judge. (3) Cross-agent consistency re-scored under the cover-bid stress test. |
| `b1_second_stack.py` | judge and extractor only | Re-scores the 48 benchmark games with a second judge and a second claim extractor and prints the monitor table next to the original. `--judge mock --extractor mock` runs offline. |
| `b2_aware_games.py play` / `score` | agents, then judge and extractor | Two new conditions in which bidders are told how the monitors work (`dial_norule_aware`, `explicit_aware`), played by three panel families and scored with the frozen monitors next to the saved baseline games. |
| `b_check_endpoint.py` | one call, or six games | Checks a model key, or compares a replacement host for the judge and extractor with the saved scores. |

`models_extra.yaml` holds the extra model entries and is merged over `config/models.yaml` at run time.
`sdb_common.py` registers the two conditions and holds the shared helpers.

## Stored results (`results_B/`)

| Files | Script | Content |
|---|---|---|
| `b0_free_analyses.json`, `b0_consistency_under_stress.csv` | `b0_free_analyses.py` | the three analyses without model calls |
| `b1_second_stack__j2_haiku__j2_haiku.json`, `b1_j2_haiku_stdout.txt` | `b1_second_stack.py` | the 48 benchmark games with Claude Haiku 5.5 as judge and claim extractor, next to the paper's stack |
| `b1_second_stack__j2_gemini__j2_haiku.json`, `b1_j2_gemini_stdout.txt` | `b1_second_stack.py --judge j2_gemini --extractor j2_haiku` | the same with Gemini 3.1 Flash Lite as judge, reading the claims extracted by Haiku |
| `b2_play_stdout.txt` | `b2_aware_games.py play` | the 24 new games (saved under `results/monitor_aware/runs/`) |
| `b2_aware_summary__j2_haiku__j2_haiku.json`, `b2_aware_per_game__j2_haiku__j2_haiku.csv`, `b2_score_j2_haiku_stdout.txt` | `b2_aware_games.py score` | monitor-aware games and their saved baselines, second stack |
| `b2_aware_summary__glm__qwen.json`, `b2_aware_per_game__glm__qwen.csv`, `b2_score_glm_qwen_stdout.txt` | `b2_aware_games.py score --judge glm --extractor qwen` | the same games with the paper's own stack (GLM judge, Qwen extractor) |

Local paths in the printed outputs are replaced by `<repo>` and `<site-packages>`. The analysis folders
written by `collusion_dial.evaluate` are under `results/second_stack/` and `results/monitor_aware/`.
