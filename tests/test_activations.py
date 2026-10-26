import numpy as np
import pytest
from intrinsic_dim.activations import final_token_indices,select_final_tokens,layer_mapping

def test_final_token_both_padding():
    mask=np.array([[1,1,0,0],[0,0,1,1],[0,1,0,1]])
    assert final_token_indices(mask).tolist()==[1,3,3]
    h=np.arange(3*4*2).reshape(3,4,2)
    np.testing.assert_equal(select_final_tokens(h,mask),h[np.arange(3),[1,3,3]])
    with pytest.raises(ValueError): final_token_indices([[0,0]])
    with pytest.raises(ValueError): final_token_indices([[1,2]])
    with pytest.raises(ValueError): select_final_tokens(h,[[1]])

def test_layer_mapping():
    mapping=layer_mapping(28)
    assert mapping[0]['kind']=='embedding'
    assert mapping[28]['post_final_norm']
    assert len(mapping)==29
