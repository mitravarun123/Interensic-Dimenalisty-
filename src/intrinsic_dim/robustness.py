import numpy as np
from .dataset import CATEGORIES
from .estimators import matrix

def preprocess(x,mode):
    x=matrix(x)
    if mode=='raw': return x
    if mode=='unit':
        norms=np.linalg.norm(x,axis=1,keepdims=True)
        if np.any(norms==0): raise ValueError('Unit normalization undefined for zero vectors')
        return x/norms
    if mode=='standardized':
        y=x-x.mean(axis=0); sd=y.std(axis=0,ddof=1)
        return np.divide(y,sd,out=np.zeros_like(y),where=sd>0)
    raise ValueError('Unknown preprocessing mode')

def group_indices(rows,split='all'):
    return {c:np.array([i for i,r in enumerate(rows) if r['task_category']==c and (split=='all' or r['split']==split)],dtype=int) for c in CATEGORIES}

def common_sample(groups,seed,count=None):
    n=min(map(len,groups.values())) if count is None else count
    if any(len(v)<n for v in groups.values()): raise ValueError('Requested sample count exceeds category count')
    rng=np.random.default_rng(seed)
    return {c:np.sort(rng.choice(v,n,replace=False)) for c,v in groups.items()}

def length_match(rows,groups,seed,bin_width=4):
    if any(rows[i]['token_count'] is None for v in groups.values() for i in v):
        return {}, {'status':'token_counts_unavailable'}
    bins={c:{} for c in groups}; rng=np.random.default_rng(seed)
    for c,indices in groups.items():
        for i in indices: bins[c].setdefault(rows[i]['token_count']//bin_width,[]).append(i)
    shared=set.intersection(*(set(v) for v in bins.values()))
    result={c:[] for c in groups}
    for b in sorted(shared):
        n=min(len(bins[c][b]) for c in groups)
        for c in groups: result[c].extend(rng.choice(bins[c][b],n,replace=False).tolist())
    return {c:np.array(sorted(v),dtype=int) for c,v in result.items()}, {'status':'ok' if shared else 'no_common_length_bins','bin_width':bin_width,'shared_bins':sorted(shared),'matched_count':len(next(iter(result.values()))),'warning':'Bin matching leaves within-bin residual length differences.'}

def family_count_control(rows,groups,matched,seed):
    """Equal retained family counts to the length-matched subset, without a length filter."""
    rng=np.random.default_rng(seed);result={}
    for category,indices in groups.items():
        counts={}
        for i in matched[category]:counts[rows[i]['template_id']]=counts.get(rows[i]['template_id'],0)+1
        chosen=[]
        for family,count in sorted(counts.items()):
            pool=np.array([i for i in indices if rows[i]['template_id']==family])
            chosen.extend(rng.choice(pool,count,replace=False).tolist())
        result[category]=np.array(sorted(chosen),dtype=int)
    return result
