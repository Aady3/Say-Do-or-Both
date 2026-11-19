"""Transfer of the frozen monitors to the UNSCRIPTED panel games only (72 ladder games),
and an index-only relabelling. Reads saved runs and scores; makes no model calls.
Usage: python addon_analysis/panel_unscripted.py [REPO_ROOT]"""
import os, sys, json, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
R = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(HERE)
sys.path.insert(0, R)
from collusion_dial.metrics.collusion import load_run
from collusion_dial.monitors.claims import score_claims
from collusion_dial.monitors.features import FAMILIES, pair_features, run_features
from collusion_dial.monitors.hybrid import fit_full_predict, fit_predict_loso
from collusion_dial.evaluate import youden_threshold
from sklearn.metrics import roc_auc_score

P = R + '/results/model_panel/analysis/'
runs = pd.read_csv(P + 'panel_runs.csv')
bench = pd.read_csv(P + 'panel/features.csv')            # the 48 benchmark games
ms = pd.read_csv(P + 'panel/monitor_scores.csv')
NOINT = FAMILIES['say'] + FAMILIES['do'] + FAMILIES['consistency']

# 1. features of the panel games, recomputed from the saved runs and cached claims
feats = {}
for d in glob.glob(R + '/results/model_panel/runs/*/*/'):
    meta, rounds = load_run(d)
    if not rounds: continue
    cp = sorted(glob.glob(os.path.join(d, 'monitor', 'claims__*.json')))
    claims = json.load(open(cp[0]))['items'] if cp else None
    scored = score_claims(claims, rounds, meta['firms']) if claims is not None else None
    feats[meta['run_id']] = run_features(pair_features(meta, rounds, scored, claims))
other = pd.DataFrame([{'run_id': r, **feats[r]} for r in runs.run_id])

# 2. sanity: reproduce the saved frozen scores of the repository's classifiers on the panel
for name, cols in (('hybrid', sum(FAMILIES.values(), [])), ('hybrid_do', FAMILIES['do'])):
    sc = fit_full_predict(bench, other, cols)
    saved = ms[ms.monitor == name].set_index('run_id').score.reindex(runs.run_id).to_numpy()
    print('max abs diff vs saved', name, float(np.nanmax(np.abs(sc - saved))))

# 3. frozen classifier (no statement-style family) and its threshold
loso, _ = fit_predict_loso(bench, NOINT)
thr = youden_threshold(bench.label, loso)
runs['score_clf'] = fit_full_predict(bench, other, NOINT); runs['flag_clf'] = (runs.score_clf >= thr).astype(float)
loso_do, _ = fit_predict_loso(bench, FAMILIES['do']); thr_do = youden_threshold(bench.label, loso_do)
runs['score_clfbid'] = fit_full_predict(bench, other, FAMILIES['do']); runs['flag_clfbid'] = (runs.score_clfbid >= thr_do).astype(float)
runs = runs.rename(columns={'score_hybrid': 'score_clfstyle', 'flag_hybrid': 'flag_clfstyle'})
print('thresholds', round(thr, 4), round(thr_do, 4))

MON = ['think', 'clf', 'clfbid', 'structured', 'saydo', 'consistency', 'say', 'do', 'clfstyle', 'screens']
def table(sub, label, name):
    print('==', name, 'games', len(sub), 'cartel', int(sub[label].sum()))
    out = {}
    for m in MON:
        s = sub[[label, 'score_' + m, 'flag_' + m]].dropna()
        c, o = s[s[label] == 1], s[s[label] == 0]
        out[m] = dict(n=len(s), auroc=roc_auc_score(s[label], s['score_' + m]), cartel=(int(c['flag_' + m].sum()), len(c)), other=(int(o['flag_' + m].sum()), len(o)))
        print(f"{m:11s} n={len(s):2d} auroc={out[m]['auroc']:.3f} cartel {out[m]['cartel'][0]}/{out[m]['cartel'][1]}={c['flag_'+m].mean():.2f} other {out[m]['other'][0]}/{out[m]['other'][1]}={o['flag_'+m].mean():.2f}")
    return out
res = {}
res['all90'] = table(runs, 'cartel', 'all 90 panel games (with the 18 scripted controls)')
u = runs[runs.condition.str.startswith('dial')].copy()
res['unscripted72'] = table(u, 'cartel', 'unscripted ladder, 72 games')
# 4. index-only label: drop the "at least two executed agreements" route, which shares its input with the consistency check
cut = u[u.cartel == 0].collusion_index.max()
u['cartel_idx'] = ((u.cartel == 1) & (u.collusion_index > cut)).astype(int)
print('unscripted cartels:', int(u.cartel.sum()), '| by index alone:', int(u.cartel_idx.sum()), '| only through executed agreements:', int(u.cartel.sum() - u.cartel_idx.sum()))
res['unscripted72_index_label'] = table(u, 'cartel_idx', 'unscripted ladder, index-only label')
# 5. highest-sister-bid baseline and cheap talk in the wild
hb = {}
for d in glob.glob(R + '/results/model_panel/runs/*/*/'):
    meta, rounds = load_run(d)
    b = [r['bids'].get(x) for r in rounds if r['valid'] for x in meta['sisters'] if r['bids'].get(x) is not None]
    hb[meta['run_id']] = max(b) if b else np.nan
runs['hb'] = runs.run_id.map(hb); u['hb'] = u.run_id.map(hb)
print('highest-sister-bid baseline AUROC: all 90 = %.3f, unscripted 72 = %.3f' % (roc_auc_score(runs.cartel, runs.hb), roc_auc_score(u.cartel, u.hb)))
ct = u[(u.cartel == 0) & (u.n_messages >= 10)]
print('cheap talk in the wild:', len(ct), {m: (int(ct['flag_' + m].dropna().sum()), int(ct['flag_' + m].notna().sum())) for m in ('say', 'saydo', 'consistency', 'clf')})
json.dump({k: {m: {kk: (float(vv) if not isinstance(vv, tuple) else list(vv)) for kk, vv in v.items()} for m, v in t.items()} for k, t in res.items()}, open(os.path.join(HERE, 'panel_unscripted.json'), 'w'), indent=1)

# 6. benchmark facts quoted in the paper (48 labelled games)
A = R + '/results/extended/analysis/eval_pooled/'
br = pd.read_csv(A + 'runs.csv'); fl = pd.read_csv(A + 'flags.csv'); th = json.load(open(A + 'thresholds.json'))
print('benchmark calls parsed:', int(br.calls_ok.sum()), 'of', int(br.calls_total.sum()), '| valid rounds:', br.valid_rounds.value_counts().to_dict())
print(br.groupby('case_type')[['pair_win_share', 'mean_overcharge']].mean().round(3).to_string())
for m in ('say', 'consistency'):
    c = fl[(fl.monitor == m) & (fl.case_type == 'cheap_talk')]
    print(m, 'cheap talk: scores', sorted(c.score.round(2)), '| leave-one-seed-out flags', int(c.flag.sum()), 'of', len(c),
          '| at frozen threshold', round(th[m], 3), ':', int((c.score >= th[m]).sum()), 'of', len(c))
    e = fl[(fl.monitor == m) & (fl.case_type == 'explicit_collusion')]
    print(m, 'explicit rings: LOSO', int(e.flag.sum()), 'of', len(e), '| frozen', int((e.score >= th[m]).sum()), 'of', len(e))
print('Say mean score, honest games:', round(fl[(fl.monitor == 'say') & fl.case_type.isin(['honest', 'honest_nomsg'])].score.mean(), 3))
