import numpy as np
import pytest
from intrinsic_dim.model_loader import resolve_dtype
from intrinsic_dim.config import Config
from intrinsic_dim.cli import main
from intrinsic_dim.dataset import generate,save_dataset
from intrinsic_dim import activations

def test_range_safe_dtype_selection():
    assert resolve_dtype('auto','cpu')=='float32'
    assert resolve_dtype('auto','cuda',False)=='float32'  # T4
    assert resolve_dtype('auto','cuda',True)=='bfloat16'
    assert resolve_dtype('float32','cuda',True)=='float32'
    assert resolve_dtype('float16','cuda',False)=='float16'
    with pytest.raises(ValueError): resolve_dtype('bfloat16','cuda',False)

def test_programmatic_cli_preserves_exception(tmp_path):
    missing=str(tmp_path/'missing.yaml')
    with pytest.raises(FileNotFoundError): main(['audit','--config',missing])

def test_nonfinite_batch_not_checkpointed(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    from test_extraction_integration import Backbone,Tokenizer,INFO
    from intrinsic_dim.io import read_json
    class NonfiniteBackbone(Backbone):
        def __call__(self,**kwargs):
            output=super().__call__(**kwargs)
            output.hidden_states[1][0,-1,0]=float('inf')
            return output
    rows=generate(8); path=tmp_path/'prompts.jsonl';save_dataset(path,rows)
    c=Config(dataset_path=str(path),output_dir=str(tmp_path/'run'),batch_size=2,device='cpu')
    monkeypatch.setattr(activations,'load_model',lambda config:(NonfiniteBackbone(),Tokenizer(),INFO))
    with pytest.raises(ValueError,match='layer=1.*infinities=1'):
        activations.extract(c)
    meta=read_json(tmp_path/'run/activations/metadata.json')
    assert meta['completed_rows']==0 and not meta['complete']
