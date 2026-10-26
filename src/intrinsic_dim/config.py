from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml

@dataclass
class Config:
    model_id: str = 'Qwen/Qwen2.5-1.5B'
    revision: str | None = None
    device: str = 'auto'
    dtype: str = 'auto'
    batch_size: int = 4
    dataset_path: str = 'data/prompts.jsonl'
    output_dir: str = 'artifacts/qwen'
    cache_dir: str | None = None
    seed: int = 42
    layers: list[int] | None = None
    target_count: int = 200
    max_tokens: int = 512
    split: str = 'evaluation'
    preprocessing: list[str] = field(default_factory=lambda: ['raw','unit','standardized'])
    repeats: int = 5
    sample_sizes: list[int] = field(default_factory=lambda: [8,16,32])
    bootstrap: int = 200
    permutations: int = 200
    confidence: float = .95
    min_families: int = 4
    length_bin_width: int = 4
    synthetic_count: int = 160
    synthetic_width: int = 24
    synthetic_layers: int = 3
    permutation_exchangeable: bool = False
    figure_dpi: int = 600
    dataset_design: str = 'legacy'
    require_preflight: bool = False
    min_evaluation_samples: int = 64
    min_matched_samples: int = 32
    uncertainty_modes: list[str] = field(default_factory=lambda: ['raw'])

    def validate(self):
        if not isinstance(self.figure_dpi,int) or not 100 <= self.figure_dpi <= 1200: raise ValueError('figure_dpi must be an integer between 100 and 1200')
        if self.dataset_design not in ['legacy','v2']: raise ValueError('Unknown dataset design')
        if self.min_evaluation_samples<3 or self.min_matched_samples<3: raise ValueError('Preflight sample minima must be >=3')
        if set(self.uncertainty_modes)-set(self.preprocessing): raise ValueError('uncertainty_modes must be selected preprocessing modes')
        if self.device not in ['auto','cpu','cuda']: raise ValueError('device must be auto, cpu, or cuda')
        if self.dtype not in ['auto','float16','float32','bfloat16']: raise ValueError('invalid dtype')
        if self.split not in ['exploratory','evaluation','all']: raise ValueError('invalid split')
        if min(self.batch_size,self.target_count,self.max_tokens,self.repeats,self.length_bin_width) < 1: raise ValueError('counts must be positive')
        if self.bootstrap < 0 or self.permutations < 0: raise ValueError('resampling counts must be nonnegative')
        if not 0 < self.confidence < 1: raise ValueError('confidence must be in (0,1)')
        if not self.preprocessing or set(self.preprocessing)-{'raw','unit','standardized'}: raise ValueError('unknown preprocessing')
        if self.layers is not None and (not self.layers or len(set(self.layers))!=len(self.layers) or min(self.layers)<0): raise ValueError('layers must be unique nonnegative indices')
        if self.min_families < 2: raise ValueError('min_families must be >=2')
        if any(n<3 for n in self.sample_sizes): raise ValueError('sample sizes must be >=3')
        return self

    def to_dict(self): return asdict(self)

def load_config(path, **overrides):
    values = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(values) - set(Config.__dataclass_fields__)
    if unknown: raise ValueError(f'Unknown config keys: {sorted(unknown)}')
    values.update({k:v for k,v in overrides.items() if v is not None})
    return Config(**values).validate()
