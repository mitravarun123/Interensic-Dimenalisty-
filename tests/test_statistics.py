from types import SimpleNamespace
import numpy as np
import pytest
from intrinsic_dim.statistics import fdr,cluster_indices,compare_clouds

def test_fdr():
    np.testing.assert_allclose(fdr([.01,.04,.03,None],method='bh')[:3],[.03,.04,.04])
    assert np.isnan(fdr([None])[0])
    assert (fdr([.01,.04],method='by')>=fdr([.01,.04],method='bh')).all()
    with pytest.raises(ValueError): fdr([1.1])

def test_cluster_resampling():
    labels=['a','a','b','b','b','c']; rng=np.random.default_rng(3)
    idx=cluster_indices(labels,rng)
    for family in set(labels):
        counts=[np.sum(idx==i) for i,v in enumerate(labels) if v==family]
        assert len(set(counts))==1

def test_bootstrap_reproducible_and_undefined():
    rng=np.random.default_rng(4); a=rng.normal(size=(24,5)); b=rng.normal(size=(24,5)); labels=np.repeat(list('abcd'),6)
    c=SimpleNamespace(min_families=4,bootstrap=25,permutations=25,permutation_exchangeable=True,confidence=.95)
    x=compare_clouds(a,b,labels,labels,'raw',c,12); y=compare_clouds(a,b,labels,labels,'raw',c,12)
    assert x==y
    assert x[0]['ci_low'] is not None and 0<x[0]['p_value']<=1
    assert x[-1]['ci_low'] is None
    c.permutation_exchangeable=False
    z=compare_clouds(a,b,labels,labels,'raw',c,12)
    assert z[0]['p_value'] is None
    c.min_families=5
    assert compare_clouds(a,b,labels,labels,'raw',c,12)[0]['ci_low'] is None
