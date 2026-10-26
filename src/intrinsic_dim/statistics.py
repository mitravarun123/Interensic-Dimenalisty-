"""Resample entire template families; estimates are cloud-level statistics."""
import numpy as np
from .estimators import measure,participation_ratio,effective_rank,spectrum_values
from .robustness import preprocess

METRICS=['pr','erank','pca_80','pca_90','pca_95','pca_99','two_nn']

def fdr(pvalues,method='by'):
    """BY default protects against arbitrary layer/metric dependence; BH optional."""
    p=np.array([np.nan if v is None else v for v in pvalues],dtype=float)
    if np.any((p[np.isfinite(p)]<0)|(p[np.isfinite(p)]>1)): raise ValueError('p values outside [0,1]')
    q=np.full(len(p),np.nan); valid=np.flatnonzero(np.isfinite(p)); m=len(valid)
    if not m:return q
    if method not in ['bh','by']: raise ValueError('Unknown FDR method')
    order=valid[np.argsort(p[valid])]; factor=np.sum(1/np.arange(1,m+1)) if method=='by' else 1.
    adjusted=p[order]*m/np.arange(1,m+1)*factor
    q[order]=np.minimum(1,np.minimum.accumulate(adjusted[::-1])[::-1]); return q

def cluster_indices(labels,rng):
    labels=np.asarray(labels); families=np.unique(labels)
    return np.concatenate([np.flatnonzero(labels==f) for f in rng.choice(families,len(families),replace=True)])

def values(x,mode):
    result,_=measure(preprocess(x,mode))
    return {m:r['estimate'] for m,r in result.items()}

def resample_values(x,mode):
    """Exact covariance spectrum via smaller Gram matrix for resampling.

    Primary measurements retain economy SVD and its rank diagnostics. Gram
    eigenvalues are used only for bootstrap functionals; numerical negatives
    are validated/clipped, and no rank inference is made from them.
    """
    x=preprocess(x,mode);x=x-x.mean(axis=0);n,d=x.shape
    if n>=d:return {m:r['estimate'] for m,r in measure(x,neighborhoods=False)[0].items()}
    eigenvalues,_=spectrum_values(np.linalg.eigvalsh((x@x.T)/(n-1))[::-1])
    result={'pr':participation_ratio(eigenvalues)['estimate'],'erank':effective_rank(eigenvalues)['estimate']}
    total=eigenvalues.sum();cv=np.cumsum(eigenvalues/total) if total>0 else None
    result.update({f'pca_{int(t*100)}':min(len(eigenvalues),int(np.searchsorted(cv,t))+1) if cv is not None else None for t in [.8,.9,.95,.99]})
    return result

def compare_clouds(a,b,labels_a,labels_b,mode,config,seed):
    """a=reasoning, b=facts. Percentile intervals; permutation is opt-in.

    Pooled family permutation requires exchangeability of family clouds under
    the null (stronger than equality of PR). Natural templates do not establish
    this, so no p value is produced by default. Bootstrap TWO-NN is disabled:
    resampling creates duplicate points, invalidating nearest-neighbor geometry.
    """
    rng=np.random.default_rng(seed); va=values(a,mode); vb=values(b,mode)
    fit_a=measure(preprocess(a,mode))[0]['two_nn']; fit_b=measure(preprocess(b,mode))[0]['two_nn']
    fa=len(np.unique(labels_a)); fb=len(np.unique(labels_b)); draws={m:[] for m in METRICS}; perms={m:[] for m in METRICS}
    enough=min(fa,fb)>=config.min_families
    if enough:
        for _ in range(config.bootstrap):
            aa=a[cluster_indices(labels_a,rng)]; bb=b[cluster_indices(labels_b,rng)]
            # Refit preprocessing inside each resampling replicate.
            ra=resample_values(aa,mode);rb=resample_values(bb,mode)
            for m in ra:
                av,bv=ra[m],rb[m]
                if av is not None and bv is not None: draws[m].append(av-bv)
        if config.permutation_exchangeable and config.permutations:
            clusters=[a[np.asarray(labels_a)==f] for f in np.unique(labels_a)]+[b[np.asarray(labels_b)==f] for f in np.unique(labels_b)]
            for _ in range(config.permutations):
                order=rng.permutation(len(clusters)); aa=np.concatenate([clusters[i] for i in order[:fa]]); bb=np.concatenate([clusters[i] for i in order[fa:]])
                pa=values(aa,mode); pb=values(bb,mode)
                for m in METRICS:
                    if pa[m] is not None and pb[m] is not None: perms[m].append(pa[m]-pb[m])
    results=[]; alpha=(1-config.confidence)/2
    for m in METRICS:
        estimate=va[m]-vb[m] if va[m] is not None and vb[m] is not None else None
        valid_draws=draws[m]; valid_perms=perms[m]
        ci=np.quantile(valid_draws,[alpha,1-alpha]).tolist() if len(valid_draws)>=max(20,int(.8*config.bootstrap)) and config.bootstrap else [None,None]
        p=(1+np.sum(np.abs(valid_perms)>=abs(estimate)))/(1+len(valid_perms)) if estimate is not None and len(valid_perms)>=max(20,int(.8*config.permutations)) and config.permutations else None
        flags=[]
        if not enough: flags.append('insufficient_template_families')
        if not config.permutation_exchangeable: flags.append('family_exchangeability_not_assumed_no_p_value')
        if m=='two_nn':
            flags.append('bootstrap_disabled_duplicates_invalidate_geometry')
            if fit_a['status']!='ok' or fit_b['status']!='ok':
                flags.append('unstable_or_undefined_neighborhood_estimate'); p=None
        if len(valid_draws)<config.bootstrap and m!='two_nn' and enough: flags.append('invalid_bootstrap_replicates')
        results.append({'estimator':m,'difference_reasoning_minus_facts':estimate,'ratio_reasoning_to_facts':va[m]/vb[m] if va[m] is not None and vb[m] is not None and vb[m]>0 else None,'effect_size_kind':'raw_dimension_difference','ci_low':ci[0],'ci_high':ci[1],'p_value':p,'bootstrap_valid':len(valid_draws),'permutations_valid':len(valid_perms),'families_reasoning':fa,'families_facts':fb,'sample_count_reasoning':len(a),'sample_count_facts':len(b),'status':'ok' if estimate is not None else 'undefined','flags':flags,'confidence':config.confidence,'sampling_unit':'template_family'})
    return results
