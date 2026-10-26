import json
from pathlib import Path
from intrinsic_dim.config import Config
from intrinsic_dim.activations import synthetic,open_cache
from intrinsic_dim.analysis import analyze
from intrinsic_dim.cli import export_project

def test_end_to_end(tmp_path):
    c=Config(output_dir=str(tmp_path/'run'),synthetic_count=64,synthetic_width=10,synthetic_layers=2,preprocessing=['raw'],split='all',bootstrap=0,permutations=0,repeats=2,sample_sizes=[8],figure_dpi=100)
    synthetic(c); result=analyze(c)
    summary=json.loads((tmp_path/'run/results/results_summary.json').read_text())
    assert result['source']=='synthetic_validation'
    assert summary['hypothesis_supported'] is None
    assert not summary['scientific_model_findings_available']
    assert len(list((tmp_path/'run/figures').glob('*.png')))==7
    assert (tmp_path/'run/results/estimates.csv').exists()
    assert all(r['outcome']=='synthetic_validation_only' for r in summary['primary_outcomes'])
    import zipfile
    path=export_project(tmp_path/'test.zip',c.output_dir)
    with zipfile.ZipFile(path) as z:
        assert not any(p.endswith('.npy') for p in z.namelist())
        assert any(p.endswith('results_summary.json') for p in z.namelist())
