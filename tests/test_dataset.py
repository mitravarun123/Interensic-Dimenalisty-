import pytest
from intrinsic_dim.dataset import generate,validate,audit
from intrinsic_dim.robustness import common_sample,group_indices,length_match

def test_generation_schema_reproducibility():
    a=generate(); assert a==generate(); assert len(a)==200
    assert audit(a)['exact_normalized_duplicates']==0
    assert {v['count'] for v in audit(a)['categories'].values()}=={50}
    for category in ['facts','arithmetic','reasoning','coding']:
        e={r['template_id'] for r in a if r['split']=='evaluation' and r['task_category']==category}
        x={r['template_id'] for r in a if r['split']=='exploratory' and r['task_category']==category}
        assert not e&x
    with pytest.raises(ValueError): validate(a+[a[0]])
    bad=[dict(r) for r in a]; del bad[0]['prompt_id']
    with pytest.raises(ValueError): validate(bad)

def test_sampling_matching():
    rows=generate(80); g=group_indices(rows)
    assert all((common_sample(g,5)[k]==common_sample(g,5)[k]).all() for k in g)
    assert length_match(rows,g,2)[1]['status']=='token_counts_unavailable'
    for i,r in enumerate(rows): r['token_count']=10+i%5
    matched,diag=length_match(rows,g,2,1)
    assert diag['status']=='ok'
    assert len({len(v) for v in matched.values()})==1
