"""Counterfactual cover-bid stress test (see README.md in this folder).
Usage: python addon_analysis/stress.py [REPO_ROOT]   (REPO_ROOT defaults to the parent of this folder)"""
import os, sys, json, glob, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
R=sys.argv[1] if len(sys.argv)>1 else os.path.dirname(HERE)
sys.path.insert(0,R)
from collusion_dial.monitors.hybrid import _model
from collusion_dial.monitors.features import FAMILIES
from collusion_dial.monitors.screens import compute_screens
from collusion_dial.metrics.collusion import load_run
from sklearn.metrics import roc_auc_score
A=R+'/results/extended/analysis/eval_pooled/'
f=pd.read_csv(A+'features.csv')
dirs={}
for d in glob.glob(R+'/results/main/runs/eval/*/')+glob.glob(R+'/results/extended/runs/eval_b2/*/'):
    meta,rounds=load_run(d); dirs[meta['run_id']]=(meta,rounds)
assert set(f.run_id)<=set(dirs), (len(dirs), set(f.run_id)-set(dirs))
DO=FAMILIES['do']
def do_feats(meta,rounds,pair):
    i,j=pair.split('-'); valid=[r for r in rounds if r['valid']]
    s=compute_screens(valid,meta['firms'],meta['reserve'])[pair]
    bi=[r['bids'].get(i) for r in valid]; bj=[r['bids'].get(j) for r in valid]
    both=[(a,b) for a,b in zip(bi,bj) if a is not None and b is not None]; corr=0.0
    if len(both)>=3:
        a,b=np.array(both).T
        if a.std()>0 and b.std()>0: corr=float(np.corrcoef(a,b)[0,1])
    allb=[b for b in bi+bj if b is not None]
    return dict(S1=s['S1'],S2=s['S2'],S3=s['S3'],pair_win_share=float(np.mean([r['winner'] in (i,j) for r in valid])),bid_corr=corr,
                round_number_bids=float(np.mean([abs(b/5-round(b/5))<1e-9 for b in allb])))
# sanity: reproduce saved do-features
err=0
for _,row in f.iterrows():
    meta,rounds=dirs[row.run_id]; d=do_feats(meta,rounds,row.pair)
    err=max(err,max(abs(d[k]-row[k]) for k in DO))
print('max abs diff vs saved do-features:',err)
# honest losing margins
marg=[]
for rid,(meta,rounds) in dirs.items():
    if meta['condition'] in ('honest','honest_nomsg'):
        for r in rounds:
            if r['valid'] and r['winning_bid'] is not None:
                marg+=[b-r['winning_bid'] for k,b in r['bids'].items() if b is not None and k!=r['winner']]
marg=np.array(marg); print('honest losing margins: n=%d median=%.1f q10=%.1f q90=%.1f'%(len(marg),np.median(marg),*np.percentile(marg,[10,90])))
# scripted cover margins before perturbation
cov=[]
for rid,(meta,rounds) in dirs.items():
    if meta['condition'] in ('explicit_collusion','tacit_collusion'):
        for r in rounds:
            if r['valid'] and r['winning_bid'] is not None:
                cov+=[r['bids'][s]-r['winning_bid'] for s in meta['sisters'] if r['bids'].get(s) is not None and s!=r['winner']]
cov=np.array(cov); print('scripted sister losing margins: n=%d median=%.1f'%(len(cov),np.median(cov)))
def perturb(meta,rounds,alpha,rng):
    out=[]
    for r in rounds:
        r=dict(r); r['bids']=dict(r['bids'])
        if r['valid'] and r['winning_bid'] is not None:
            for s in meta['sisters']:
                b=r['bids'].get(s)
                if b is None or s==r['winner']: continue
                tgt=r['winning_bid']+max(0.5,float(rng.choice(marg)))
                if b>tgt: r['bids'][s]=round(b-alpha*(b-tgt),1)
        out.append(r)
    return out
def maxsis(meta,rounds):
    return max(r['bids'][s] for r in rounds if r['valid'] for s in meta['sisters'] if r['bids'].get(s) is not None)
lab=f.label.to_numpy(int); seeds=f.seed.to_numpy()
print('one-feature baseline (max sister bid) original AUROC: %.3f'%roc_auc_score(lab,[maxsis(*dirs[r]) for r in f.run_id]))
NOINT=FAMILIES['say']+FAMILIES['do']+FAMILIES['consistency']
def loso(train,test,cols):
    sc=np.full(len(test),np.nan)
    for s in sorted(set(seeds)):
        tr,te=seeds!=s,seeds==s
        m=_model().fit(np.nan_to_num(train.loc[tr,cols].to_numpy(float)),lab[tr])
        sc[te]=m.predict_proba(np.nan_to_num(test.loc[te,cols].to_numpy(float)))[:,1]
    return sc
rows=[]
for alpha in (0,0.25,0.5,0.75,1.0):
    acc={k:[] for k in ('base','bid_frozen','clf_frozen','bid_refit','clf_refit','rec_clf_frozen','covmed')}
    for rs in range(20):
        rng=np.random.default_rng(100+rs); g=f.copy(); base=[]; cm=[]
        for n,(rid,pair,l) in enumerate(zip(f.run_id,f.pair,f.label)):
            meta,rounds=dirs[rid]
            if l==1 and alpha>0:
                rounds=perturb(meta,rounds,alpha,rng)
                for k,v in do_feats(meta,rounds,pair).items(): g.loc[n,k]=v
            if l==1:
                cm+=[r['bids'][s]-r['winning_bid'] for r in rounds if r['valid'] and r['winning_bid'] is not None for s in meta['sisters'] if r['bids'].get(s) is not None and s!=r['winner']]
            base.append(maxsis(meta,rounds))
        acc['base'].append(roc_auc_score(lab,base)); acc['covmed'].append(np.median(cm))
        s1=loso(f,g,DO); s2=loso(f,g,NOINT)
        acc['bid_frozen'].append(roc_auc_score(lab,s1)); acc['clf_frozen'].append(roc_auc_score(lab,s2))
        acc['rec_clf_frozen'].append(float((s2[lab==1]>=0.5).mean()))
        acc['bid_refit'].append(roc_auc_score(lab,loso(g,g,DO))); acc['clf_refit'].append(roc_auc_score(lab,loso(g,g,NOINT)))
        if alpha==0: break
    rows.append(dict(alpha=alpha,**{k:float(np.mean(v)) for k,v in acc.items()},**{k+'_sd':float(np.std(v)) for k,v in acc.items()}))
    print(rows[-1])
pd.DataFrame(rows).to_csv(os.path.join(HERE,'stress.csv'),index=False)
