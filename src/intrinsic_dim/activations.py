"""Resumable, per-layer float32 NPY memmaps, atomic batch checkpoints."""
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from .io import write_json,read_json,fingerprint,versions,source_fingerprint
from .dataset import load_dataset,save_dataset,audit,generate
from .model_loader import load_model

def final_token_indices(attention_mask):
    mask=np.asarray(attention_mask)
    if mask.ndim!=2 or not np.isin(mask,[0,1]).all() or np.any(mask.sum(axis=1)==0): raise ValueError('Attention mask must be binary 2D with a valid token in every row')
    return np.max(np.where(mask==1,np.arange(mask.shape[1]),-1),axis=1)

def select_final_tokens(hidden,attention_mask):
    """Works for left and right padding, including noncontiguous valid positions."""
    idx=final_token_indices(attention_mask)
    if hidden.ndim!=3 or tuple(hidden.shape[:2])!=tuple(np.shape(attention_mask)): raise ValueError('Hidden state/mask shape mismatch')
    try:
        import torch
        if isinstance(hidden,torch.Tensor): return hidden[torch.arange(len(idx),device=hidden.device),torch.as_tensor(idx,device=hidden.device)]
    except ImportError: pass
    return hidden[np.arange(len(idx)),idx]

def layer_mapping(num_blocks):
    # Qwen2 hidden_states[final] includes final RMSNorm; earlier entries are
    # residual-stream inputs to the next block. Final raw block hook is separate.
    return [{'layer':i,'kind':'embedding' if i==0 else 'transformer_block','label':'Embedding output' if i==0 else f'Block {i}'+(' + final RMSNorm' if i==num_blocks else ''),'post_final_norm':i==num_blocks} for i in range(num_blocks+1)]

def create_arrays(directory,layers,shape,resume=False):
    arrays={}
    for layer in layers:
        path=directory/f'layer_{layer:03d}.npy'
        arrays[layer]=np.load(path,mmap_mode='r+') if resume else np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=shape)
        if arrays[layer].shape!=shape or arrays[layer].dtype!=np.float32: raise ValueError(f'Invalid cache array {path}')
    return arrays

def extract(config):
    try:
        import torch
    except ImportError as e:
        raise RuntimeError('Real extraction requires PyTorch: pip install -e \".[model]\"') from e
    if config.require_preflight:
        from .preflight import prepare_dataset
        rows,_=prepare_dataset(config)
    else:
        rows=load_dataset(config.dataset_path)
    model,tokenizer,info=load_model(config)
    layers=config.layers if config.layers is not None else list(range(info['transformer_blocks']+1))
    if max(layers)>info['transformer_blocks']: raise ValueError('Layer index exceeds available hidden-state tuple')
    n=len(rows); width=info['hidden_width']; directory=Path(config.output_dir)/'activations'; directory.mkdir(parents=True,exist_ok=True)
    for r in rows:
        r['token_count']=len(tokenizer(r['prompt'],add_special_tokens=True)['input_ids'])
        r['token_count_method']=f"{info['tokenizer_class']}:{info['tokenizer_name']}"
        if r['token_count']>config.max_tokens: raise ValueError(f"Prompt {r['prompt_id']} exceeds max_tokens; increase limit. Truncation is disabled.")
    cache_key=fingerprint({'rows':rows,'model':info,'layers':layers,'seed':config.seed,'max_tokens':config.max_tokens})
    meta_path=directory/'metadata.json'; resume=meta_path.exists()
    if resume:
        meta=read_json(meta_path)
        if meta['cache_key']!=cache_key: raise ValueError('Cache differs from current dataset/model/config; choose a new output directory')
        start=meta['completed_rows']
    else:
        start=0
        meta={'cache_key':cache_key,'source':'model','model':info,'config':config.to_dict(),'software':versions(),'sample_ids':[r['prompt_id'] for r in rows],'rows':rows,'layers':[m for m in layer_mapping(info['transformer_blocks']) if m['layer'] in layers],'shape':[n,width],'storage_dtype':'float32','completed_rows':0,'complete':False,'estimated_storage_bytes':n*width*len(layers)*4,'source_fingerprint':source_fingerprint(),'dataset_fingerprint':fingerprint(rows),'convention':'Final non-padding input token; prompt only, no generation; standard HF hidden_states tuple, final output after RMSNorm.'}
        write_json(meta_path,meta)
    arrays=create_arrays(directory,layers,(n,width),resume)
    print(f"Actual device: {info['device']}; dtype: {info['dtype']}; activation storage {meta['estimated_storage_bytes']/2**20:.1f} MiB; resuming at {start}/{n}")
    try:
        with torch.inference_mode():
            for offset in tqdm(range(start,n,config.batch_size),desc='Extract prompt hidden states'):
                batch=rows[offset:offset+config.batch_size]
                tokens=tokenizer([r['prompt'] for r in batch],return_tensors='pt',padding=True,truncation=False,add_special_tokens=True)
                masks=tokens['attention_mask'].cpu().numpy(); final_token_indices(masks)
                tokens={k:v.to(info['device']) for k,v in tokens.items()}
                outputs=model(**tokens,output_hidden_states=True,use_cache=False,return_dict=True)
                if len(outputs.hidden_states)!=info['transformer_blocks']+1: raise ValueError('Unexpected layer tuple length')
                for layer in layers:
                    values=select_final_tokens(outputs.hidden_states[layer],masks).float().cpu().numpy()
                    if values.shape!=(len(batch),width):
                        raise ValueError(f'Hidden representation shape mismatch: layer={layer}, expected={(len(batch),width)}, got={values.shape}')
                    if not np.isfinite(values).all():
                        bad=np.flatnonzero(~np.isfinite(values).all(axis=1))
                        ids=[batch[i]['prompt_id'] for i in bad]
                        raise ValueError(f'Nonfinite hidden representation: layer={layer}, prompt_ids={ids}, dtype={info["dtype"]}, device={info["device"]}, NaNs={int(np.isnan(values).sum())}, infinities={int(np.isinf(values).sum())}. No batch checkpoint was committed. Retry the entire extraction with dtype=float32, batch_size=1 and a fresh output directory; never mix precision modes or replace invalid values with zeros.')
                    arrays[layer][offset:offset+len(batch)]=values; arrays[layer].flush()
                meta['completed_rows']=offset+len(batch); meta['complete']=meta['completed_rows']==n
                write_json(meta_path,meta)
                del outputs,tokens
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError('GPU out of memory. Checkpoint retained; resume with smaller --batch-size (try 1).') from e
    save_dataset(Path(config.output_dir)/'datasets'/'tokenized_prompts.jsonl',rows)
    write_json(Path(config.output_dir)/'datasets'/'audit.json',audit(rows))
    return meta

def synthetic(config):
    """Same null dimensionality distribution for each category; validation only."""
    if config.synthetic_count<40 or config.synthetic_count%4: raise ValueError('synthetic_count must be >=40 and divisible by four')
    if config.dataset_design=='v2':
        rows=load_dataset(config.dataset_path)
    else:
        rows=generate(config.synthetic_count,config.seed)
        for r in rows:
            r['token_count']=len(r['prompt'].split()); r['token_count_method']='synthetic_whitespace_proxy_not_model_tokens'
    rng=np.random.default_rng(config.seed)
    directory=Path(config.output_dir)/'activations'; directory.mkdir(parents=True,exist_ok=True)
    layers=list(range(config.synthetic_layers)); width=config.synthetic_width; n=len(rows)
    arrays=create_arrays(directory,layers,(n,width))
    for layer in layers:
        latent=min(width,4+layer*2)
        basis=np.linalg.qr(rng.normal(size=(width,latent)))[0]
        arrays[layer][:]=(rng.normal(size=(n,latent))@basis.T + .02*rng.normal(size=(n,width))).astype(np.float32)
        arrays[layer].flush()
    meta={'source':'synthetic_validation','model':{'model_id':None,'resolved_revision':None,'device':'cpu','dtype':'float32'},'software':versions(),'config':config.to_dict(),'sample_ids':[r['prompt_id'] for r in rows],'rows':rows,'layers':[{'layer':l,'label':f'Synthetic layer {l}','kind':'synthetic'} for l in layers],'shape':[n,width],'complete':True,'completed_rows':n,'source_fingerprint':source_fingerprint(),'dataset_fingerprint':fingerprint(rows),'convention':'Synthetic Gaussian clouds; no pretrained model used','known_latent_dimensions':[min(width,4+l*2) for l in layers],'cache_key':fingerprint(config.to_dict())}
    write_json(directory/'metadata.json',meta); save_dataset(Path(config.output_dir)/'datasets'/'synthetic_prompts.jsonl',rows); write_json(Path(config.output_dir)/'datasets'/'audit.json',audit(rows))
    return meta

def open_cache(config):
    directory=Path(config.output_dir)/'activations'; meta=read_json(directory/'metadata.json')
    if not meta.get('complete') or meta['completed_rows']!=meta['shape'][0]: raise ValueError('Extraction incomplete; resume extract first')
    if meta['sample_ids']!=[r['prompt_id'] for r in meta['rows']]: raise ValueError('Sample ID alignment mismatch')
    return directory,meta
