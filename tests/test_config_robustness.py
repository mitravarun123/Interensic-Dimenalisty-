import numpy as np
import pytest
from intrinsic_dim.config import Config,load_config
from intrinsic_dim.robustness import preprocess,common_sample
from intrinsic_dim.estimators import pca_spectrum,effective_rank

def test_config_reject_unknown_and_invalid(tmp_path):
    p=tmp_path/'bad.yaml';p.write_text('typo: 1')
    with pytest.raises(ValueError,match='Unknown'):load_config(p)
    for c in [Config(device='mps'),Config(layers=[1,1]),Config(preprocessing=['uncentered']),Config(confidence=1),Config(sample_sizes=[1])]:
        with pytest.raises(ValueError):c.validate()

def test_preprocessing_and_invalid_data():
    x=np.array([[1.,2,5],[3.,2,7],[4.,2,9]])
    unit=preprocess(x,'unit')
    np.testing.assert_allclose(np.linalg.norm(unit,axis=1),1)
    standardized=preprocess(x,'standardized')
    np.testing.assert_allclose(standardized[:,1],0)
    np.testing.assert_allclose(standardized.mean(axis=0),0,atol=1e-15)
    np.testing.assert_allclose(standardized[:,[0,2]].std(axis=0,ddof=1),1)
    with pytest.raises(ValueError):preprocess(np.zeros((3,2)),'unit')
    with pytest.raises(ValueError):pca_spectrum([[float('nan'),1],[2,3]])
    with pytest.raises(ValueError):common_sample({'a':[0],'b':[1]},3,2)

def test_known_pca_threshold_and_entropy():
    # Centered, orthogonal columns with covariance eigenvalues 9 and 1.
    x=np.array([[1.,1],[1.,-1],[-1.,1],[-1.,-1]])*np.sqrt(3/4)*np.array([3.,1])
    p=pca_spectrum(x)
    np.testing.assert_allclose(p['eigenvalues'],[9,1])
    assert p['pca_dimensions']['0.8']==1
    assert p['pca_dimensions']['0.95']==2
    expected=np.exp(-.9*np.log(.9)-.1*np.log(.1))
    assert effective_rank([9,1])['estimate']==pytest.approx(expected)
