"""Holm and Bonferroni correction of the seven delta tests (see README.md in this folder).
Usage: python addon_analysis/holm.py [REPO_ROOT]   (REPO_ROOT defaults to the parent of this folder)"""
import os, sys, json, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
R=sys.argv[1] if len(sys.argv)>1 else os.path.dirname(HERE)
sys.path.insert(0,R)
from collusion_dial.monitors.hybrid import fit_predict_loso
from collusion_dial.monitors.features import FAMILIES
from sklearn.metrics import roc_auc_score
A=R+'/results/extended/analysis/eval_pooled/'
m=pd.read_csv(A+'monitor_scores.csv'); f=pd.read_csv(A+'features.csv')
piv=m[m.stake.isna()].pivot_table(index='run_id',columns='monitor',values='score')
f=f.set_index('run_id').loc[piv.index].reset_index()
lab=f.label.to_numpy(int)
noint=FAMILIES['say']+FAMILIES['do']+FAMILIES['consistency']
sc_no,_=fit_predict_loso(f,noint); sc_do,_=fit_predict_loso(f,FAMILIES['do'])
print('repro hybrid_do', np.abs(sc_do-piv['hybrid_do'].to_numpy()).max())
S={'say':piv['say'],'do':piv['do'],'saydo':piv['saydo'],'structured':piv['structured'],'clf_bid':sc_do,'clf':sc_no,'hybrid':piv['hybrid'],'consistency':piv['consistency'],'think':piv['think']}
S={k:np.asarray(v,float) for k,v in S.items()}
for k,v in S.items(): print(k, round(roc_auc_score(lab,v),4))
targets=['saydo','structured','clf_bid','clf','hybrid','think','consistency']
pos,neg=np.where(lab==1)[0],np.where(lab==0)[0]
def boot(n,seed):
    rng=np.random.default_rng(seed); out={t:[] for t in targets}
    for _ in range(n):
        idx=np.concatenate([rng.choice(pos,len(pos)),rng.choice(neg,len(neg))])
        a={c:roc_auc_score(lab[idx],S[c][idx]) for c in S}
        best=max(a['say'],a['do'])
        for t in targets: out[t].append(a[t]-best)
    return {t:np.asarray(v) for t,v in out.items()}
res={}
for n,seed in ((2000,0),(20000,1)):
    b=boot(n,seed); print('--- n_boot',n)
    ps={t:float((b[t]<=0).mean()) for t in targets}
    order=sorted(targets,key=lambda t:ps[t]); holm={}; run=0
    for i,t in enumerate(order):
        run=max(run,min(1,ps[t]*(len(targets)-i))); holm[t]=run
    for t in targets:
        pt=roc_auc_score(lab,S[t])-max(roc_auc_score(lab,S['say']),roc_auc_score(lab,S['do']))
        lo,hi=np.percentile(b[t],[2.5,97.5]); alo,ahi=np.percentile(b[t],[100*0.025/7,100*(1-0.025/7)])
        print(f"{t:12s} d={pt:+.3f} CI95=[{lo:+.3f},{hi:+.3f}] p={ps[t]:.4f} holm={holm[t]:.4f} bonfCI=[{alo:+.3f},{ahi:+.3f}]")
        res[(n,t)]=dict(d=pt,lo=lo,hi=hi,p=ps[t],holm=holm[t],alo=alo,ahi=ahi)
# classifier vs its bid-only version
b=boot(20000,1); d=b['clf']-b['clf_bid']; print('clf - clf_bid', np.percentile(d,[2.5,97.5]), (d<=0).mean())
json.dump({f"{n}_{t}":v for (n,t),v in res.items()},open(os.path.join(HERE,'holm.json'),'w'),indent=1)
