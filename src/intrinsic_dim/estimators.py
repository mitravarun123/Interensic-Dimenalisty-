"""Float64 full-spectrum estimators; covariance uses N-1 normalization."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

def matrix(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape)<1 or not np.isfinite(x).all():
        raise ValueError('Expected a nonempty finite N x d matrix')
    return x

def spectrum_values(eigenvalues):
    e = np.asarray(eigenvalues, dtype=np.float64)
    if e.ndim != 1 or not len(e) or not np.isfinite(e).all(): raise ValueError('Invalid spectrum')
    tolerance = np.finfo(float).eps * max(1,len(e)) * max(float(np.max(np.abs(e))), np.finfo(float).tiny) * 10
    if np.any(e < -tolerance): raise ValueError('Substantial negative covariance eigenvalue')
    return np.maximum(e,0), int(np.sum(e<0))

def participation_ratio(eigenvalues):
    e, clipped = spectrum_values(eigenvalues)
    scale = e.max()
    if scale == 0: return {'estimate': None, 'status':'zero_variance','clipped_negative':clipped,'spectrum':'full'}
    e = e/scale
    return {'estimate': float(e.sum()**2 / (e@e)), 'status':'ok','clipped_negative':clipped,'spectrum':'full'}

def effective_rank(eigenvalues):
    e, clipped = spectrum_values(eigenvalues)
    if e.max()==0: return {'estimate':None,'status':'zero_variance','entropy':None,'normalized_entropy':None,'positive_rank':0}
    e = e/e.max(); p=e/e.sum(); p=p[p>0]
    entropy = float(-np.sum(p*np.log(p))); r=len(p)
    return {'estimate': float(np.exp(entropy)), 'entropy':entropy,'normalized_entropy':entropy/np.log(r) if r>1 else 0.,'positive_rank':r,'status':'ok','clipped_negative':clipped}

def pca_spectrum(x, rank_rtol=None):
    x = matrix(x); n,d=x.shape
    if n<2: raise ValueError('PCA requires at least two samples')
    x = x-x.mean(axis=0)
    # Economy SVD avoids a d x d covariance allocation and Gram cancellation.
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    e=s*s/(n-1); total=float(e.sum())
    ratios=e/total if total>0 else np.zeros_like(e)
    tol=(max(n,d)*np.finfo(float).eps if rank_rtol is None else rank_rtol)*s[0]
    if rank_rtol is not None and rank_rtol<0: raise ValueError('rank_rtol must be nonnegative')
    rank=int(np.count_nonzero(s>tol))
    cumulative=np.cumsum(ratios)
    thresholds={str(t): min(len(e),int(np.searchsorted(cumulative,t))+1) if total>0 else None for t in [.8,.9,.95,.99]}
    return {'eigenvalues':e,'singular_values':s,'explained_variance':ratios,'cumulative':cumulative,'pca_dimensions':thresholds,'numerical_rank':rank,'singular_tolerance':float(tol),'sample_rank_bound':min(n-1,d),'total_variance':total,'condition_number_nonzero':float((s[0]/s[rank-1])**2) if rank else None,'feature_variance_min':float(np.var(x,axis=0,ddof=1).min()),'feature_variance_max':float(np.var(x,axis=0,ddof=1).max()),'status':'ok' if total>0 else 'zero_variance'}

def two_nn(x):
    """Facco-style empirical-CDF origin fit, omitting largest 10% ratios.

    F_i=(i-.5)/M; y=-log(1-F); x=log(mu); slope is dimension.
    This finite-sample plotting-position variant is explicitly documented.
    Duplicate points invalidate the cloud; they are never silently removed.
    """
    x=matrix(x); n=len(x)
    if n<10: return {'estimate':None,'status':'insufficient_samples','sample_count':n}
    distances,_=cKDTree(x).query(x,k=3,workers=1)
    r1,r2=distances[:,1],distances[:,2]
    if np.any(r1<=0): return {'estimate':None,'status':'duplicate_points','duplicate_count':int(np.sum(r1<=0))}
    mu=r2/r1
    if not np.isfinite(mu).all() or np.any(mu<1): return {'estimate':None,'status':'invalid_ratios'}
    log_mu=np.sort(np.log(mu)); keep=int(np.floor(.9*n))
    xx=log_mu[:keep]; yy=-np.log1p(-(np.arange(n)+.5)/n)[:keep]
    denominator=xx@xx
    if denominator<=np.finfo(float).tiny: return {'estimate':None,'status':'degenerate_distances'}
    slope=float(xx@yy/denominator); residual=yy-slope*xx
    r2fit=1-float(residual@residual)/float(np.sum((yy-yy.mean())**2))
    return {'estimate':slope,'status':'ok' if r2fit>=.8 else 'poor_fit','r_squared':r2fit,'cdf_rmse':float(np.sqrt(np.mean(residual**2))),'fit_log_mu':xx,'fit_log_survival':yy,'ratios':mu,'fit_count':keep,'metric':'euclidean','tail_discard_fraction':.1}

def measure(x, neighborhoods=True):
    p=pca_spectrum(x); pr=participation_ratio(p['eigenvalues']); er=effective_rank(p['eigenvalues'])
    result={'pr':pr,'erank':er, **{f'pca_{int(float(k)*100)}':{'estimate':v,'status':p['status']} for k,v in p['pca_dimensions'].items()}}
    if neighborhoods: result['two_nn']=two_nn(x)
    return result,p

def distance_diagnostics(x, max_points=500):
    x=matrix(x)[:max_points]; distances=pdist(x)
    return {'point_count':len(x),'quantiles':np.quantile(distances,[0,.1,.5,.9,1]).tolist() if len(distances) else [],'coefficient_of_variation':float(distances.std()/distances.mean()) if len(distances) and distances.mean()>0 else None}
