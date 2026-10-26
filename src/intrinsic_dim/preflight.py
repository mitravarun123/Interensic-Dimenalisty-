"""Check actual tokenizer lengths and family coverage before loading weights."""
from pathlib import Path
from .dataset import load_dataset,save_dataset,audit,CATEGORIES
from .robustness import group_indices,length_match
from .io import write_json,fingerprint,read_json,dataset_input_fingerprint

def preflight(rows,config):
    groups=group_indices(rows,'evaluation');matched,matching=length_match(rows,groups,config.seed,config.length_bin_width)
    details={};errors=[]
    for c in CATEGORIES:
        idx=groups[c]; mids=matched.get(c,[])
        families=len({rows[i]['template_id'] for i in idx}); mf=len({rows[i]['template_id'] for i in mids})
        details[c]={'evaluation_samples':len(idx),'evaluation_families':families,'length_matched_samples':len(mids),'length_matched_families':mf}
        if len(idx)<config.min_evaluation_samples:errors.append(f'{c}: evaluation samples {len(idx)} < {config.min_evaluation_samples}')
        if families<config.min_families:errors.append(f'{c}: evaluation families {families} < {config.min_families}')
    if matching['status']!='ok':errors.append('No verified common model-token length bins across the four categories')
    if matching.get('matched_count',0)<config.min_matched_samples:errors.append(f'Length-matched sample count < {config.min_matched_samples}')
    for c in ['reasoning','facts']:
        if details[c]['length_matched_families']<config.min_families:errors.append(f'{c}: matched family coverage below {config.min_families}')
    return {'status':'passed' if not errors else 'failed','errors':errors,'design_versions':sorted({r.get('design_version','legacy') for r in rows}),'counts':details,'length_matching':matching,'dataset_fingerprint':fingerprint(rows),'warnings':['Matching is within tokenizer bins, not exact continuous equality.','Matched category family coverage differs; primary CI uses reasoning/facts only.','Independent family sampling and absence of crossed entity dependence remain assumptions.','Human ground-truth/ambiguity review is still required; passing preflight does not establish scientific validity.']}

def prepare_dataset(config):
    try:from transformers import AutoTokenizer
    except ImportError as e:raise RuntimeError('prepare requires tokenizer dependencies: pip install -e ".[model]"') from e
    rows=load_dataset(config.dataset_path)
    manifest_path=Path(config.dataset_path).parent/'design_manifest_v2.json'
    if config.dataset_design=='v2' and manifest_path.exists() and Path(config.dataset_path).name=='prompts_v2.jsonl':
        manifest=read_json(manifest_path)
        if manifest['dataset_input_fingerprint']!=dataset_input_fingerprint(rows):
            raise ValueError('Frozen v2 dataset inputs changed. Preserve the run and explicitly document/refreeze the revised design before extraction.')
    tokenizer=AutoTokenizer.from_pretrained(config.model_id,revision=config.revision,cache_dir=config.cache_dir,trust_remote_code=False)
    method=f'{type(tokenizer).__name__}:{tokenizer.name_or_path}'
    for row in rows:
        row['token_count']=len(tokenizer(row['prompt'],add_special_tokens=True)['input_ids']);row['token_count_method']=method
        if row['token_count']>config.max_tokens:raise ValueError(f"Prompt {row['prompt_id']} exceeds max_tokens; truncation is disabled")
    result=preflight(rows,config)
    result['tokenizer']={'model_id':config.model_id,'requested_revision':config.revision,'class':type(tokenizer).__name__,'padding_side':tokenizer.padding_side}
    path=Path(config.output_dir)/'datasets';path.mkdir(parents=True,exist_ok=True)
    save_dataset(path/'prepared_prompts.jsonl',rows);write_json(path/'preflight.json',result);write_json(path/'audit.json',audit(rows))
    print('Dataset preflight:',result['status']);print(result['counts'])
    if config.require_preflight and result['errors']:raise ValueError('Dataset design preflight failed BEFORE loading weights: '+ '; '.join(result['errors']))
    return rows,result
