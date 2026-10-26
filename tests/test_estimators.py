import numpy as np
import pytest
from intrinsic_dim.estimators import participation_ratio as pr,effective_rank as er,pca_spectrum,two_nn

def test_uniform_concentrated():
    assert pr([1,1,1])['estimate']==3
    assert er([1,1,1])['estimate']==pytest.approx(3)
    assert pr([9,1])['estimate']==pytest.approx(100/82)
    assert er([1,0,0])['estimate']==1
    assert er([1,0,0])['normalized_entropy']==0

def test_scale_zero_negative():
    for f in [pr,er]:
        assert f([3,2,1])['estimate']==pytest.approx(f(np.array([3,2,1])*1e200)['estimate'])
        assert f([0,0])['estimate'] is None
        assert f([1,-1e-16])['estimate']==pytest.approx(1)
        with pytest.raises(ValueError): f([1,-.01])

def test_spectrum_covariance_rank():
    x=np.random.default_rng(3).normal(size=(7,25))
    p=pca_spectrum(x)
    expected=np.linalg.eigvalsh(np.cov(x,rowvar=False))[::-1][:7]
    np.testing.assert_allclose(p['eigenvalues'],expected,atol=1e-12)
    assert p['numerical_rank']==6
    assert p['explained_variance'].sum()==pytest.approx(1)
    np.testing.assert_allclose(p['eigenvalues'],pca_spectrum(x+100)['eigenvalues'],atol=1e-12)
    assert pca_spectrum(np.diag([1.,1e-10,0]),rank_rtol=1e-6)['numerical_rank']==1
    assert pca_spectrum(np.ones((5,9)))['pca_dimensions']['0.9'] is None

def test_two_nn_dimensions_duplicates():
    rng=np.random.default_rng(72)
    for d in [2,5]:
        result=two_nn(rng.normal(size=(4000,d)))
        assert abs(result['estimate']-d)<d*.25
        assert result['r_squared']>.9
    assert two_nn(np.ones((20,3)))['status']=='duplicate_points'
    assert two_nn(rng.normal(size=(5,3)))['status']=='insufficient_samples'
