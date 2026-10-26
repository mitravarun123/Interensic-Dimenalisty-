import argparse
from pathlib import Path
import zipfile
from .config import Config, load_config
from .dataset import generate,save_dataset,load_dataset,audit
from .io import write_json

def export_project(destination,output_dir=None,include_activations=False):
    root=Path(__file__).resolve().parents[2]; destination=Path(destination).resolve(); destination.parent.mkdir(parents=True,exist_ok=True)
    excluded={'.git','.venv','__pycache__','.pytest_cache','.cache','.ipynb_checkpoints','build','dist'}
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(root.rglob('*')):
            rel=p.relative_to(root)
            if not p.is_file() or any(part in excluded or part.endswith('.egg-info') for part in rel.parts) or p.suffix in ['.zip','.pyc'] or p==destination: continue
            if 'artifacts' in rel.parts: continue
            archive.write(p,Path('intrinsic-dimension-transformers')/rel)
        if output_dir:
            output=Path(output_dir).resolve()
            for p in sorted(output.rglob('*')):
                if not p.is_file() or p.suffix=='.zip' or p==destination: continue
                if not include_activations and p.suffix=='.npy':continue
                archive.write(p,Path('intrinsic-dimension-transformers')/'artifacts'/'exported_run'/p.relative_to(output))
    return destination

def main(argv=None):
    parser=argparse.ArgumentParser(description='Task-conditional intrinsic dimension research pipeline')
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ['generate','audit','prepare','extract','analyze','report','smoke-test','export']:
        p=sub.add_parser(command)
        p.add_argument('--config',default='configs/default.yaml')
        p.add_argument('--device',choices=['auto','cpu','cuda'])
        p.add_argument('--batch-size',type=int); p.add_argument('--model-id'); p.add_argument('--revision'); p.add_argument('--dataset-path'); p.add_argument('--output-dir'); p.add_argument('--seed',type=int); p.add_argument('--target-count',type=int)
        p.add_argument('--layers',type=lambda s:[int(v) for v in s.split(',')]); p.add_argument('--split',choices=['all','exploratory','evaluation'])
        if command=='export': p.add_argument('--destination',default='intrinsic-dimension-transformers.zip'); p.add_argument('--include-activations',action='store_true')
    args=parser.parse_args(argv); values=vars(args)
    try:
        config=load_config(args.config,**{k:v for k,v in values.items() if k in Config.__dataclass_fields__})
        if args.command=='generate':
            from .dataset_v2 import generate_v2
            generator=generate_v2 if config.dataset_design=='v2' else generate
            rows=generator(config.target_count,config.seed); save_dataset(config.dataset_path,rows); print(f'Generated {len(rows)} prompts: {config.dataset_path}')
        elif args.command=='audit':
            rows=load_dataset(config.dataset_path); output=Path(config.output_dir)/'datasets'/'audit.json'; write_json(output,audit(rows)); print(output)
        elif args.command=='prepare':
            from .preflight import prepare_dataset
            prepare_dataset(config)
        elif args.command=='extract':
            from .activations import extract
            extract(config)
        elif args.command=='analyze':
            from .analysis import analyze
            analyze(config)
        elif args.command=='report':
            from .reporting import report
            report(config); print(Path(config.output_dir)/'results'/'research_report.md')
        elif args.command=='smoke-test':
            from .activations import synthetic
            from .analysis import analyze
            synthetic(config); analyze(config); print('Synthetic end-to-end validation complete; no model findings.')
        elif args.command=='export':print(export_project(args.destination,config.output_dir,args.include_activations))
    except (RuntimeError,ValueError,FileNotFoundError,ImportError,MemoryError) as e:
        if argv is not None:
            raise
        parser.exit(1,f'Error: {e}\n')

if __name__=='__main__': main()
