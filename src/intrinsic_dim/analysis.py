"""Layer-streamed analysis with source data, controls, and explicit failures."""
from pathlib import Path
from dataclasses import replace
from itertools import combinations
import numpy as np
from tqdm.auto import tqdm
from .activations import open_cache
from .estimators import measure,distance_diagnostics
from .robustness import preprocess,group_indices,common_sample,length_match,family_count_control
from .statistics import compare_clouds,fdr
from .io import write_json,write_csv,fingerprint,source_fingerprint


def analyze(config):
    directory,meta=open_cache(config); rows=meta['rows']; results=Path(config.output_dir)/'results'; results.mkdir(parents=True,exist_ok=True)
    if config.require_preflight:
        from .preflight import preflight
        check=preflight(rows,config);write_json(results/'preflight.json',check)
        if check['errors']:raise ValueError('Analysis preflight failed: '+ '; '.join(check['errors']))
    groups=group_indices(rows,config.split)
    if min(map(len,groups.values()))<3: raise ValueError('Analysis needs >=3 prompts in every category for the selected split')
    primary=common_sample(groups,config.seed)
    matched,matching=length_match(rows,groups,config.seed,config.length_bin_width)
    variants={'sample_matched':primary}
    if matched and min(map(len,matched.values()))>=3:
        variants['length_matched']=matched
        variants['length_count_control']=common_sample(groups,config.seed+50000,len(next(iter(matched.values()))))
        variants['length_family_control']=family_count_control(rows,groups,matched,config.seed+60000)
    for split in ['exploratory','evaluation']:
        g=group_indices(rows,split)
        if min(map(len,g.values()))>=3: variants['template_'+split]=common_sample(g,config.seed)
    estimates=[]; comparisons=[]; diagnostics=[]; spectra=[]; robustness=[]; projections=[]
    selected_layers=meta['layers'] if config.layers is None else [l for l in meta['layers'] if l['layer'] in config.layers]
    if config.layers is not None and len(selected_layers)!=len(config.layers): raise ValueError('Selected layers absent from activation cache')
    for layer in tqdm(selected_layers,desc='Analyze layers'):
        l=layer['layer']; comparison_cache={}; full=np.load(directory/f'layer_{l:03d}.npy',mmap_mode='r')
        if list(full.shape)!=meta['shape'] or not np.isfinite(full).all(): raise ValueError(f'Invalid layer {l} array')
        for mode in config.preprocessing:
            for variant,g in variants.items():
                clouds={}
                for category,idx in g.items():
                    x=preprocess(full[idx],mode); clouds[category]=x
                    metrics,p=measure(x)
                    base={'layer':l,'layer_label':layer['label'],'task_category':category,'sample_count':len(idx),'preprocessing':mode,'seed':config.seed,'subset':variant,'split':config.split if variant in ['sample_matched','length_matched','length_count_control','length_family_control'] else variant.removeprefix('template_'),'source':meta['source']}
                    for metric,r in metrics.items(): estimates.append({**base,'estimator':metric,'estimate':r['estimate'],'status':r['status']})
                    diag={**base,'numerical_rank':p['numerical_rank'],'rank_bound':p['sample_rank_bound'],'singular_tolerance':p['singular_tolerance'],'total_variance':p['total_variance'],'condition_number_nonzero':p['condition_number_nonzero'],'feature_variance_min':p['feature_variance_min'],'feature_variance_max':p['feature_variance_max'],'distances':distance_diagnostics(x),'two_nn':metrics.get('two_nn'),'sample_ids':[rows[i]['prompt_id'] for i in idx]}
                    diagnostics.append(diag)
                    if variant=='sample_matched':
                        for j,(e,ev,cv,s) in enumerate(zip(p['eigenvalues'],p['explained_variance'],p['cumulative'],p['singular_values'])):
                            spectra.append({**base,'component':j+1,'eigenvalue':float(e),'explained_variance':float(ev),'cumulative':float(cv),'singular_value':float(s)})
                aa=g['reasoning']; bb=g['facts']
                # Pass raw clouds; preprocessing must be recomputed per replicate.
                comparison_config=config if mode in config.uncertainty_modes else replace(config,bootstrap=0,permutations=0)
                key=(mode,tuple(aa.tolist()),tuple(bb.tolist()))
                if key not in comparison_cache:
                    comparison_cache[key]=compare_clouds(np.asarray(full[aa]),np.asarray(full[bb]),[rows[i]['template_id'] for i in aa],[rows[i]['template_id'] for i in bb],mode,comparison_config,config.seed+l)
                comp=comparison_cache[key]
                for r in comp: comparisons.append({**r,'layer':l,'layer_label':layer['label'],'preprocessing':mode,'subset':variant,'seed':config.seed,'analysis_role':'primary' if layer.get('kind')!='embedding' and mode=='raw' and variant=='sample_matched' and r['estimator']=='pr' else 'secondary_or_robustness','split':config.split if variant in ['sample_matched','length_matched','length_count_control','length_family_control'] else variant.removeprefix('template_'),'source':meta['source']})
            for n in config.sample_sizes:
                if n>min(map(len,groups.values())):
                    robustness.append({'layer':l,'preprocessing':mode,'sample_count':n,'status':'insufficient_samples','source':meta['source']}); continue
                for rep in range(config.repeats):
                    seed=config.seed+10000+rep; g=common_sample(groups,seed,n)
                    for category,idx in g.items():
                        metrics,_=measure(preprocess(full[idx],mode))
                        for metric,r in metrics.items(): robustness.append({'layer':l,'layer_label':layer['label'],'preprocessing':mode,'task_category':category,'sample_count':n,'seed':seed,'repeat':rep,'estimator':metric,'estimate':r['estimate'],'status':r['status'],'source':meta['source']})
        # Fit only exploratory prompts; apply the same centered basis to both splits.
        train=np.array([i for i,r in enumerate(rows) if r['split']=='exploratory']); train_x=np.asarray(full[train],dtype=float)
        mean=train_x.mean(axis=0); _,_,vt=np.linalg.svd(train_x-mean,full_matrices=False); basis=vt[:2].T
        coordinates=(np.asarray(full,dtype=float)-mean)@basis
        for i,r in enumerate(rows): projections.append({'layer':l,'layer_label':layer['label'],'prompt_id':r['prompt_id'],'task_category':r['task_category'],'split':r['split'],'pc1':float(coordinates[i,0]),'pc2':float(coordinates[i,1]),'fit_split':'exploratory','source':meta['source']})
        del full
    # Separate pre-specified primary layer family from secondary/robustness family.
    for role in ['primary','secondary_or_robustness']:
        indices=[i for i,r in enumerate(comparisons) if r['analysis_role']==role]
        q=fdr([comparisons[i]['p_value'] for i in indices],method='by')
        for i,value in zip(indices,q): comparisons[i]['adjusted_p_value']=float(value) if np.isfinite(value) else None; comparisons[i]['fdr_method']='Benjamini-Yekutieli'; comparisons[i]['fdr_family']=role
    # All task pairs are descriptive; inference is restricted to primary contrast.
    pairwise=[]
    keys=sorted({(r['layer'],r['preprocessing'],r['subset'],r['estimator']) for r in estimates})
    for l,mode,subset,metric in keys:
        cloud={r['task_category']:r for r in estimates if (r['layer'],r['preprocessing'],r['subset'],r['estimator'])==(l,mode,subset,metric)}
        for a,b in combinations(sorted(cloud),2):
            ra,rb=cloud[a],cloud[b]; av,bv=ra['estimate'],rb['estimate']
            pairwise.append({'layer':l,'layer_label':ra['layer_label'],'preprocessing':mode,'subset':subset,'estimator':metric,'task_a':a,'task_b':b,'difference_a_minus_b':av-bv if av is not None and bv is not None else None,'ratio_a_to_b':av/bv if av is not None and bv is not None and bv>0 else None,'sample_count_a':ra['sample_count'],'sample_count_b':rb['sample_count'],'status_a':ra['status'],'status_b':rb['status'],'inference':'descriptive_only','ci_low':None,'ci_high':None,'adjusted_p_value':None,'source':meta['source']})
    write_csv(results/'pairwise_comparisons.csv',pairwise)
    for name,data in [('estimates',estimates),('comparisons',comparisons),('spectra',spectra),('robustness',robustness),('projections',projections)]: write_csv(results/f'{name}.csv',data)
    write_json(results/'diagnostics.json',diagnostics)
    write_json(results/'selection.json',{'primary':{c:[rows[i]['prompt_id'] for i in idx] for c,idx in primary.items()},'variants':{v:{c:[rows[i]['prompt_id'] for i in idx] for c,idx in g.items()} for v,g in variants.items()},'length_matching':matching})
    bundle={'source':meta['source'],'metadata':meta,'analysis_config':config.to_dict(),'analysis_fingerprint':fingerprint({'config':config.to_dict(),'source':source_fingerprint(),'activation_cache':meta['cache_key']}),'analysis_source_fingerprint':source_fingerprint(),'estimates':estimates,'comparisons':comparisons,'pairwise_comparisons':pairwise,'spectra':spectra,'robustness':robustness,'projections':projections,'length_matching':matching,'diagnostics':diagnostics}
    write_json(results/'analysis.json',bundle)
    from .visualization import figures
    figures(bundle,Path(config.output_dir)/'figures')
    from .reporting import report
    report(config,bundle)
    return bundle
