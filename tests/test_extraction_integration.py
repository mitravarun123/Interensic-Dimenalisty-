"""Test real torch inference/cache path with a local fake backbone, no download."""
from types import SimpleNamespace
import numpy as np
import pytest
from intrinsic_dim.config import Config
from intrinsic_dim.dataset import generate,save_dataset
from intrinsic_dim import activations

torch=pytest.importorskip('torch')

class Tokenizer:
    def __call__(self,text,return_tensors=None,padding=False,truncation=False,add_special_tokens=True):
        if isinstance(text,str): return {'input_ids':list(range(1,1+len(text.split())))}
        lens=[len(t.split()) for t in text]; width=max(lens)
        ids=torch.zeros((len(text),width),dtype=torch.long)
        mask=torch.zeros_like(ids)
        for i,n in enumerate(lens): ids[i,-n:]=torch.arange(1,n+1); mask[i,-n:]=1
        return {'input_ids':ids,'attention_mask':mask}

class Backbone:
    def __init__(self,fail_call=None): self.calls=0; self.fail_call=fail_call
    def __call__(self,input_ids,attention_mask,output_hidden_states,use_cache,return_dict):
        assert not torch.is_grad_enabled()
        assert output_hidden_states and not use_cache
        self.calls+=1
        if self.calls==self.fail_call: raise RuntimeError('simulated interruption')
        h=input_ids.float().unsqueeze(-1).repeat(1,1,3)
        return SimpleNamespace(hidden_states=(h,h+1,h+2))

INFO={'model_id':'local_test_backbone','requested_revision':None,'resolved_revision':'fixture','device':'cpu','dtype':'float32','tokenizer_class':'fixture','tokenizer_name':'fixture','hidden_width':3,'transformer_blocks':2}

def test_extract_resume_and_cache_mismatch(tmp_path,monkeypatch):
    rows=generate(16); path=tmp_path/'prompts.jsonl'; save_dataset(path,rows)
    c=Config(dataset_path=str(path),output_dir=str(tmp_path/'run'),device='cpu',batch_size=3)
    model=Backbone(fail_call=2)
    monkeypatch.setattr(activations,'load_model',lambda config:(model,Tokenizer(),INFO))
    with pytest.raises(RuntimeError,match='interruption'): activations.extract(c)
    from intrinsic_dim.io import read_json
    meta_path=tmp_path/'run/activations/metadata.json'
    assert read_json(meta_path)['completed_rows']==3
    with pytest.raises(ValueError,match='incomplete'): activations.open_cache(c)
    model=Backbone(); c.batch_size=2
    monkeypatch.setattr(activations,'load_model',lambda config:(model,Tokenizer(),INFO))
    meta=activations.extract(c)
    assert meta['complete'] and meta['completed_rows']==16
    layer=np.load(tmp_path/'run/activations/layer_002.npy')
    np.testing.assert_allclose(layer[:,0],[len(r['prompt'].split())+2 for r in rows])
    assert len(meta['sample_ids'])==16
    before=model.calls; activations.extract(c); assert model.calls==before
    c.seed=17
    with pytest.raises(ValueError,match='Cache differs'): activations.extract(c)
