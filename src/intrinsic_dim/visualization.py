"""Figures derive exclusively from cached measurements; PNG/PDF + source CSV."""
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path(tempfile.gettempdir())/'intrinsic_dim_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from .dataset import CATEGORIES
from .io import write_csv,write_json

COLORS={'facts':'#0072B2','arithmetic':'#E69F00','reasoning':'#009E73','coding':'#CC79A7'}
METRICS=['pr','erank','pca_90','two_nn']
LABELS={'pr':'Participation ratio','erank':'Covariance effective rank','pca_90':'PCA components (90% variance)','two_nn':'TWO-NN dimension'}

def figures(bundle,directory):
    directory.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':12, 'axes.labelsize':12, 'axes.titlesize':13, 'xtick.labelsize':11, 'ytick.labelsize':11, 'lines.linewidth':2, 'savefig.facecolor':'white', 'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'figure.dpi':110})
    tag='SYNTHETIC VALIDATION — no model findings' if bundle['source']=='synthetic_validation' else 'Prompt-only model activations'
    layers=sorted({r['layer'] for r in bundle['estimates']}); selected=layers[-1]
    base=[r for r in bundle['estimates'] if r['preprocessing']=='raw' and r['subset']=='sample_matched']
    layer_labels={r['layer']:r['layer_label'] for r in base}
    manifest=[]
    dpi=int(bundle['analysis_config'].get('figure_dpi',600))
    def save(fig,name,source):
        fig.suptitle(tag,fontsize=14, fontweight='semibold'); fig.tight_layout(rect=[0,0,1,.96],pad=1.5)
        for ax in fig.axes:
            ax.tick_params(length=4)
            if 'index' in ax.get_xlabel().lower() or 'layer' in ax.get_xlabel().lower(): ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for ext in ['png','pdf','svg']: fig.savefig(directory/f'{name}.{ext}',bbox_inches='tight',dpi=dpi if ext=='png' else 300)
        write_csv(directory/f'{name}_source.csv',source); plt.close(fig); manifest.append(name)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    source=[r for r in base if r['layer']==selected]
    for ax,m in zip(axes.flat,METRICS):
        data={r['task_category']:r for r in source if r['estimator']==m}
        for j,c in enumerate(CATEGORIES):
            r=data.get(c,{})
            if r.get('estimate') is not None: ax.bar(j,r['estimate'],color=COLORS[c])
            else: ax.text(j,0,'undefined',rotation=90,va='bottom')
        ax.set_xticks(range(4),CATEGORIES); ax.set_ylabel(LABELS[m]); ax.set_title(layer_labels[selected])
    save(fig,'01_dimensionality_by_task',source)
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    for ax,m in zip(axes.flat,METRICS):
        for c in CATEGORIES:
            data=sorted([r for r in base if r['estimator']==m and r['task_category']==c],key=lambda r:r['layer'])
            ax.plot([r['layer'] for r in data],[np.nan if r['estimate'] is None else r['estimate'] for r in data],marker='o',ms=5,label=c,color=COLORS[c])
        ax.set_ylabel(LABELS[m]); ax.set_xlabel('Layer index (0 = embedding for real model)'); ax.legend(fontsize=10)
    save(fig,'02_layerwise_evolution',base)
    spectrum=[r for r in bundle['spectra'] if r['layer']==selected and r['preprocessing']=='raw']
    fig,axes=plt.subplots(1,3,figsize=(16,5.5))
    for c in CATEGORIES:
        data=[r for r in spectrum if r['task_category']==c]; k=[r['component'] for r in data]; e=[r['eigenvalue'] for r in data]
        axes[0].plot(k,e,color=COLORS[c],label=c)
        axes[1].plot(k,[v if v>0 else np.nan for v in e],color=COLORS[c]); axes[1].set_yscale('log')
        axes[2].plot(k,[r['cumulative'] for r in data],color=COLORS[c])
    for ax,title in zip(axes,['Covariance spectrum','Covariance spectrum (log)','Cumulative explained variance']): ax.set_title(title+'\n'+layer_labels[selected]); ax.set_xlabel('Component')
    axes[0].set_ylabel('Eigenvalue'); axes[1].set_ylabel('Eigenvalue'); axes[2].set_ylabel('Variance fraction'); axes[0].legend()
    save(fig,'03_pca_spectrum',spectrum)
    source=[r for r in bundle['projections'] if r['layer']==selected]
    fig,ax=plt.subplots(figsize=(10,7))
    for c in CATEGORIES:
        for split,marker in [('exploratory','o'),('evaluation','^')]:
            data=[r for r in source if r['task_category']==c and r['split']==split]
            ax.scatter([r['pc1'] for r in data],[r['pc2'] for r in data],s=35,alpha=.65,color=COLORS[c],marker=marker,label=f'{c}: {split}')
    ax.set_title(f'2D PCA projection; fit on exploratory split only\n{layer_labels[selected]} — not an ID measurement'); ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=10)
    save(fig,'04_pca_projection',source)
    source=[r for r in bundle['robustness'] if r['layer']==selected and r.get('estimator')=='pr']
    fig,axes=plt.subplots(1,len(bundle['analysis_config']['preprocessing']),figsize=(16,6),squeeze=False)
    for ax,mode in zip(axes.flat,bundle['analysis_config']['preprocessing']):
        for c in CATEGORIES:
            data=[r for r in source if r['task_category']==c and r['preprocessing']==mode and r['estimate'] is not None]; ns=sorted({r['sample_count'] for r in data})
            med=[]; lo=[]; hi=[]
            for n in ns:
                vals=[r['estimate'] for r in data if r['sample_count']==n]; med.append(np.median(vals)); lo.append(min(vals)); hi.append(max(vals))
            ax.plot(ns,med,'o-',color=COLORS[c],label=c); ax.fill_between(ns,lo,hi,color=COLORS[c],alpha=.1)
        ax.set_title(mode+'; shading = subsampling range'); ax.set_xlabel('Samples per category'); ax.set_ylabel('PR'); ax.legend(fontsize=10)
    save(fig,'05_robustness',source)
    source=[r for r in bundle['comparisons'] if r['estimator']=='pr' and r['preprocessing']=='raw']
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    for subset in sorted({r['subset'] for r in source}):
        data=sorted([r for r in source if r['subset']==subset],key=lambda r:r['layer'])
        axes[0].plot([r['layer'] for r in data],[r['difference_reasoning_minus_facts'] for r in data],'o-',label=subset)
        valid=[r for r in data if r['ci_low'] is not None]
        if valid: axes[0].fill_between([r['layer'] for r in valid],[r['ci_low'] for r in valid],[r['ci_high'] for r in valid],alpha=.1)
    axes[0].axhline(0,color='gray',ls='--'); axes[0].set_xlabel('Hidden-state tuple index'); axes[0].set_ylabel('PR: reasoning minus facts'); axes[0].legend(fontsize=10); axes[0].set_title('Length and template controls')
    for c in CATEGORIES:
        lengths=[r['token_count'] for r in bundle['metadata']['rows'] if r['task_category']==c and r['token_count'] is not None]
        if lengths: axes[1].hist(lengths,bins=15,alpha=.4,label=c,color=COLORS[c])
    axes[1].set_xlabel('Token count (whitespace proxy)' if any('whitespace_proxy' in r['token_count_method'] for r in bundle['metadata']['rows']) else 'Input token count (model tokenizer)'); axes[1].set_ylabel('Prompts'); axes[1].legend(fontsize=10); axes[1].set_title(bundle['length_matching']['status'])
    save(fig,'06_length_template_controls',source)
    # Fit diagnostic explicitly accompanies the neighborhood estimator.
    fig,ax=plt.subplots(figsize=(10,7)); fit_rows=[]
    for diag in bundle['diagnostics']:
        if diag['layer']!=selected or diag['preprocessing']!='raw' or diag['subset']!='sample_matched': continue
        t=diag['two_nn']; c=diag['task_category']
        if t.get('fit_log_mu') is None: continue
        xx=np.array(t['fit_log_mu']); yy=np.array(t['fit_log_survival'])
        ax.plot(xx,yy,'.',color=COLORS[c],label=f"{c} (R²={t['r_squared']:.2f})"); ax.plot(xx,xx*t['estimate'],color=COLORS[c])
        fit_rows.extend({'task_category':c,'log_mu':float(x),'minus_log_survival':float(y),'fitted':float(x*t['estimate'])} for x,y in zip(xx,yy))
    ax.set_xlabel('log(r₂/r₁)'); ax.set_ylabel('−log(1 − empirical CDF)'); ax.set_title('TWO-NN CDF fit diagnostics');
    if fit_rows: ax.legend(fontsize=10)
    save(fig,'07_two_nn_diagnostics',fit_rows)
    write_json(directory/'manifest.json',{'figures':manifest,'source':bundle['source'],'selected_layer':selected,'selected_layer_label':layer_labels[selected],'png_dpi':dpi,'vector_formats':['pdf','svg'],'uncertainty':'Cluster-bootstrap intervals only in comparison/control figure; subsampling ranges are descriptive.'})
