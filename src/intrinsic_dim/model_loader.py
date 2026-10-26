"""Optional model dependencies are imported only for real extraction."""
import random
import numpy as np

def resolve_dtype(requested, device, native_bfloat16=False):
    """Choose range-safe inference; T4 lacks native BF16 and uses FP32.

    Qwen checkpoints use BF16; forcing FP16 narrows exponent range. Explicit
    FP16 remains available for controlled experiments but is not auto-selected.
    """
    dtype = ('bfloat16' if device=='cuda' and native_bfloat16 else 'float32') if requested=='auto' else requested
    if device=='cpu' and dtype!='float32': raise ValueError('CPU default path requires float32')
    if dtype=='bfloat16' and device=='cuda' and not native_bfloat16:
        raise ValueError('GPU lacks native bfloat16; use float32 for range-safe extraction')
    return dtype

def load_model(config):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        raise RuntimeError('Real extraction needs model dependencies: pip install -e \".[model]\"') from e
    device='cuda' if config.device=='auto' and torch.cuda.is_available() else ('cpu' if config.device=='auto' else config.device)
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable; use --device cpu or select a GPU runtime')
    native_bf16 = device=='cuda' and torch.cuda.get_device_capability()[0]>=8 and torch.cuda.is_bf16_supported()
    dtype_name=resolve_dtype(config.dtype,device,native_bf16)
    if dtype_name=='float16':
        import warnings
        warnings.warn('Explicit float16 can overflow BF16 checkpoint representations. If nonfinite values occur, rerun the entire extraction in a fresh float32 cache.',RuntimeWarning)
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    if device=='cuda': torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    kwargs={'revision':config.revision,'cache_dir':config.cache_dir,'trust_remote_code':False}
    try:
        tokenizer=AutoTokenizer.from_pretrained(config.model_id,**kwargs)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None: raise ValueError('Tokenizer has no pad or EOS token')
            tokenizer.pad_token=tokenizer.eos_token
        # AutoModel avoids materializing vocabulary-sized logits; same hidden backbone.
        model=AutoModel.from_pretrained(config.model_id,torch_dtype=getattr(torch,dtype_name),attn_implementation='eager',**kwargs).to(device).eval()
    except Exception as e:
        raise RuntimeError(f'Could not load {config.model_id} on {device}. Check model access, network, cache, and memory. Original error: {e}') from e
    info={'model_id':config.model_id,'requested_revision':config.revision,'resolved_revision':getattr(model.config,'_commit_hash',None),'device':device,'dtype':dtype_name,'dtype_policy':'auto_native_bfloat16_else_float32' if config.dtype=='auto' else 'explicit','tokenizer_class':type(tokenizer).__name__,'tokenizer_name':tokenizer.name_or_path,'tokenizer_revision':tokenizer.init_kwargs.get('_commit_hash'),'padding_side':tokenizer.padding_side,'pad_token_id':tokenizer.pad_token_id,'eos_token_id':tokenizer.eos_token_id,'hidden_width':model.config.hidden_size,'transformer_blocks':model.config.num_hidden_layers,'deterministic_algorithms':'warn_only','attention_implementation':'eager','gpu_name':torch.cuda.get_device_name(0) if device=='cuda' else None}
    return model,tokenizer,info
