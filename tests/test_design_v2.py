import ast
import re
import numpy as np
import pytest
from intrinsic_dim.dataset_v2 import generate_v2
from intrinsic_dim.dataset import load_dataset,generate,audit
from intrinsic_dim.config import load_config,Config
from intrinsic_dim.preflight import preflight
from intrinsic_dim.statistics import resample_values
from intrinsic_dim.estimators import measure
from intrinsic_dim.robustness import preprocess

def test_v2_reproducibility_distinct_families_and_split():
    a=generate_v2();assert a==generate_v2();assert len(a)==640
    for category in ['facts','arithmetic','reasoning','coding']:
        subset=[r for r in a if r['task_category']==category]
        assert len(subset)==160
        for split in ['exploratory','evaluation']:
            s=[r for r in subset if r['split']==split]
            assert len(s)==80 and len({r['template_id'] for r in s})==10
        assert len({r['template_id'] for r in subset})==20
    assert audit(a)['exact_normalized_duplicates']==0
    assert all(r['ground_truth_source'] for r in a if r['task_category']=='facts')
    # The legacy pilot is not silently relabeled as a new experiment.
    assert {r['prompt_id'] for r in a}.isdisjoint({r['prompt_id'] for r in generate()})

def test_coding_ground_truth_on_controlled_generated_expressions():
    allowed={name:getattr(__import__('builtins'),name) for name in ['sum','max','sorted','list','zip','enumerate','all','any']}
    for row in generate_v2():
        if row['task_category']!='coding':continue
        expression=re.search(r'evaluate (.*?)\. Return',row['prompt']).group(1)
        tree=ast.parse(expression,mode='eval')
        assert not any(isinstance(n,(ast.Import,ast.ImportFrom,ast.Lambda)) for n in ast.walk(tree))
        result=eval(compile(tree,'<fixed-generated-test>','eval'),{'__builtins__':allowed},{})
        assert repr(result)==row['expected_answer']

def test_actual_tokenized_dataset_passes_preflight_and_legacy_fails():
    config=load_config('configs/default.yaml')
    rows=load_dataset(config.dataset_path)
    result=preflight(rows,config)
    assert result['status']=='passed'
    assert result['length_matching']['matched_count']>=32
    assert result['counts']['reasoning']['length_matched_families']>=8
    assert result['counts']['facts']['length_matched_families']>=8
    assert all('Qwen' in r['token_count_method'] for r in rows)
    old=generate()
    assert preflight(old,config)['status']=='failed'

def test_gram_bootstrap_metrics_agree_with_primary_svd():
    x=np.random.default_rng(5).normal(size=(30,100))
    # Include duplicate resampling rows and a constant feature.
    x[2]=x[0];x[:,5]=1
    for mode in ['raw','unit','standardized']:
        fast=resample_values(x,mode);slow=measure(preprocess(x,mode),neighborhoods=False)[0]
        for key in fast:assert fast[key]==pytest.approx(slow[key]['estimate'],rel=1e-10,abs=1e-10)

def test_v2_end_to_end_intervals(tmp_path):
    from intrinsic_dim.activations import synthetic
    from intrinsic_dim.analysis import analyze
    c=load_config('configs/design_validation.yaml',output_dir=str(tmp_path/'v2'))
    c.bootstrap=30;c.synthetic_width=128;c.synthetic_layers=2;c.preprocessing=['raw'];c.repeats=1;c.figure_dpi=100
    synthetic(c);bundle=analyze(c)
    primary=[r for r in bundle['comparisons'] if r['analysis_role']=='primary']
    assert primary and all(r['ci_low'] is not None and r['bootstrap_valid']==30 for r in primary)
    controls=[r for r in bundle['comparisons'] if r['estimator']=='pr' and r['subset']=='length_matched']
    assert controls and all(r['ci_low'] is not None for r in controls)
    assert all(r['p_value'] is None for r in primary)
    assert bundle['source']=='synthetic_validation'

def test_length_controls_preserve_counts_and_family_composition():
    from collections import Counter
    from intrinsic_dim.robustness import group_indices,length_match,family_count_control,common_sample
    rows=load_dataset('data/prompts_v2.jsonl');g=group_indices(rows,'evaluation')
    matched,_=length_match(rows,g,42,4);controlled=family_count_control(rows,g,matched,55)
    count_control=common_sample(g,66,len(matched['facts']))
    for c in g:
        assert len(controlled[c])==len(count_control[c])==len(matched[c])
        assert Counter(rows[i]['template_id'] for i in controlled[c])==Counter(rows[i]['template_id'] for i in matched[c])
        assert set(controlled[c]).issubset(set(g[c]))

def test_frozen_input_change_stops_before_tokenizer_download(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from intrinsic_dim.dataset import save_dataset
    from intrinsic_dim.io import write_json,dataset_input_fingerprint
    from intrinsic_dim.preflight import prepare_dataset
    rows=generate_v2();expected=dataset_input_fingerprint(rows)
    rows[0]['prompt'] += ' Changed after freeze.'
    path=tmp_path/'prompts_v2.jsonl';save_dataset(path,rows)
    write_json(tmp_path/'design_manifest_v2.json',{'dataset_input_fingerprint':expected})
    class NeverDownload:
        @staticmethod
        def from_pretrained(*args,**kwargs):raise AssertionError('Tokenizer must not load after authored-input mismatch')
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=NeverDownload))
    c=load_config('configs/default.yaml',dataset_path=str(path),output_dir=str(tmp_path/'run'))
    with pytest.raises(ValueError,match='Frozen v2 dataset inputs changed'):prepare_dataset(c)
