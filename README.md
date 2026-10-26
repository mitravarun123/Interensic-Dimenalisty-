# Intrinsic dimensionality across transformer tasks — version 2

A reproducible representation-geometry study using **Qwen/Qwen2.5-1.5B**, final non-padding input-token hidden states at every layer, PCA spectra, participation ratio, covariance effective rank and TWO-NN. No dimensionality ordering is assumed. No training or fine-tuning is performed.

## Corrected experimental design

The completed 200-prompt Colab experiment is a **legacy pilot**: its evaluation split had only two families/12 prompts per category and no length-matched control. Those activations are useful pilot observations, not a statistically established reasoning effect. Preserve that output separately.

The revised bank contains **640 prompts**, **160 per category**, **20 named task families/category**, with **10 families/80 prompts per category held out**. Source-linked facts cover several domains; arithmetic, explicit logic and Python literal prediction each have 20 task families. Exact Qwen-token checks verified a four-category length-matched evaluation subset of **52 prompts per category**. Retained families: facts 10, arithmetic 8, reasoning 8, coding 7. This is bin matching, not proof of perfect experimental matching or independent families.

The new notebook repeats the checks **before loading weights** and stops if evaluation count, family coverage or length matching is inadequate. The primary raw spectral comparison receives **500 family-bootstrap replicates**, conditional on independent-family resampling assumptions. Arbitrary task-label permutation p-values remain disabled; more prompts do not establish exchangeability. Reports distinguish unperformed tests, undefined estimates and expected constant-token embedding diagnostics. See [revised design and limitations](docs/design_v2.md).

## Google Colab: upload two files

1. Upload **`Qwen_Colab_V2.ipynb`** using Colab's File → Upload notebook.
2. Select Runtime → Change runtime type → GPU.
3. Run the first code cell and upload **`intrinsic-dimension-transformers-v2-colab.zip`**.
4. Run cells top to bottom. Drive storage is enabled at the user's request; authorize it when prompted. Checkpoints/results go to `MyDrive/intrinsic_dimension/qwen_v2_fp32`, preserving the old pilot.
5. The tokenizer preflight must pass; then extraction loads the pinned pretrained model in **CUDA float32, batch size 1**, as used in the successful pilot. The weight cache can be reused; activation arrays must be newly extracted for the 640-row dataset.
6. Analysis generates all layer/metric tables, raw conditional intervals and separately labeled sensitivity controls. It runs on CPU after extraction and can take time.
7. The final cell downloads **`intrinsic_dimension_v2_results.zip`** and saves a Drive copy. Preserve activations for reanalysis; weights are excluded from exports.

All seven figure types export as **600-DPI PNG plus vector PDF/SVG**, with larger fonts, readable legends and source CSVs. Notebook previews are scaled; downloaded originals retain full resolution. Use PDF/SVG for publication and zooming. `figure_dpi` is configurable (100–1200).

Colab hardware allocation varies; CUDA absence stops real extraction. Installing compatible dependencies is explicit in the setup cell. If a preloaded package changes, restart the session and rerun setup. `Qwen_Colab_Ready.ipynb` and `notebooks/colab_pipeline.ipynb` contain the same updated workflow. [Step-by-step guide](docs/colab_quickstart.md).

## Local CPU validation

Python 3.10+; the public Colab run used Python 3.13. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m intrinsic_dim.cli smoke-test --config configs/smoke_test.yaml
```

Synthetic validation needs no GPU or model download and remains visibly labeled. Optional local Jupyter: `python -m pip install -e '.[notebooks]'`. `requirements.txt` supplies a numerical pin target originally aimed at Python 3.10–3.12; the editable package's compatible ranges are preferred in current Colab/Python 3.13 to retain its CUDA PyTorch. Validation versions and limitations are in [validation](docs/validation.md).

## Real-model CLI

```bash
python -m pip install -e '.[model,test]'
python -m intrinsic_dim.cli audit --config configs/default.yaml
python -m intrinsic_dim.cli prepare --config configs/default.yaml
python -m intrinsic_dim.cli extract --config configs/default.yaml
python -m intrinsic_dim.cli analyze --config configs/default.yaml
python -m intrinsic_dim.cli report --config configs/default.yaml
python -m intrinsic_dim.cli export --config configs/default.yaml --destination research_export.zip
```

The reviewed-by-code source bank is already shipped, with exact tokenizer counts; **human ambiguity review is pending**. `prepare` recomputes lengths and stores a fingerprint/preflight. It loads only the tokenizer. Extraction repeats this gate. `generate --config configs/default.yaml` deterministically rebuilds un-tokenized version-2 prompts; run `prepare` afterward. The current fixed v2 bank supports balanced counts 160–640 in multiples of 80; extend factual fixture coverage and families for larger studies rather than duplicating rows.

Common CLI overrides: `--device cpu|cuda|auto`, `--batch-size`, `--model-id`, `--revision`, `--dataset-path`, `--output-dir`, `--seed`, `--target-count`, `--layers 0,7,14,21,28`, `--split exploratory|evaluation|all`. Run `python -m intrinsic_dim.cli --help`. A different model/tokenizer must pass its own exact-length preflight. Keep revised evaluation settings fixed before extracting new activations; the study was redesigned after the pilot.

## Precision and extraction

Model revision: `8faed761d45a263340a0528343f099c05c9a4323`. `trust_remote_code=False`; no quantization. Explicit FP32 is used in the v2 profile. Auto dtype uses native BF16 when supported, otherwise FP32; forced FP16 can overflow and is not automatic. CPU requires FP32 and substantial RAM. The 8-GB M1 machine is appropriate for validation; real extraction is recommended on Colab GPU.

Attention-mask-aware final-token indexing works for either padding side. No answers are generated. Standard HF hidden-state tuple: index 0=embedding; 1..28=block representations, with final RMSNorm included at index 28. A shared final input token has constant embedding vectors; that layer is noninformative for cross-prompt covariance geometry and is excluded from primary block inference without introducing artificial token noise.

One float32 N×d NPY memmap per layer, atomic metadata checkpoints after flushing each valid batch, preserved IDs/task labels, source/software/model/dtype/seed metadata. Same-cache compatible resume is supported; failed/nonfinite batches never advance the checkpoint. Precision/model/data/layer/seed changes require a fresh activation output. Estimated v2 activation storage is about **109 MiB**, separate from weights and transient batch tensors. Analysis streams one layer and uses per-cloud float64 economy SVD. Small Gram covariance spectra speed bootstrap functionals; rank diagnostics still use primary SVD.

## Repository and results

```text
MathHeavy/
├── Qwen_Colab_V2.ipynb
├── configs/{default,smoke_test,cpu_small}.yaml
├── data/{prompts_v2.jsonl,factual_fixtures_v2.json,preflight_v2_verified.json,...}
├── notebooks/{colab_pipeline,00_environment_setup,...,04_visualize_and_report}.ipynb
├── src/intrinsic_dim/
│   ├── dataset.py, dataset_v2.py, preflight.py, config.py
│   ├── model_loader.py, activations.py, estimators.py, statistics.py
│   └── analysis.py, robustness.py, visualization.py, reporting.py, cli.py, io.py
├── tests/, scripts/, docs/
└── artifacts/<run>/
    ├── activations/{metadata.json,layer_000.npy,...}
    ├── datasets/{preflight.json,prepared_prompts.jsonl,audit.json,...}
    ├── results/{research_report.md,results_summary.json,analysis.json,
    │           estimates.csv,comparisons.csv,pairwise_comparisons.csv,
    │           spectra.csv,robustness.csv,projections.csv,diagnostics.json,
    │           selection.json,preflight.json}
    └── figures/{01_...,02_...,...,07_...}.{png,pdf,svg} + source CSVs
```

Figures: category dimensions at the last selected layer, layer evolution, linear/log covariance spectra and cumulative variance, exploratory-fit 2D PCA projection, sample/preprocessing sensitivity, length/template controls and TWO-NN CDF diagnostics. A 2D projection is not an intrinsic-dimension measurement. All task-pair comparisons are descriptive; reasoning/facts is the specified primary contrast.

The code ZIP has no weights, activations or fabricated scientific outputs. Results ZIPs include tables/figures/metadata; arrays are excluded unless `--include-activations` is chosen. Arrays are saved separately in Drive by default in the revised notebook. Preserve the legacy pilot ZIP and its actual model outputs.

## Scientific limits

Named families do not prove independent random sampling. The curated bank remains subject to operator, domain, entity, syntax and answer-type confounds. Conditional bootstrap intervals need valid independent-unit assumptions; no calibrated significance or H1 support is manufactured. More resampling cannot fix a deficient design. TWO-NN replacement-bootstrap is disabled because duplicates invalidate neighborhoods. Human review, crossed-entity/operator controls, model replication and a justified inferential design remain follow-up needs.

Hidden-state width, numerical rank, PCA thresholds, participation ratio, covariance effective rank and local intrinsic dimension are different quantities. Higher dimensionality does not establish computational effort, reasoning ability, model correctness or causal circuits. Null/reversed/inconclusive outcomes are allowed. [Methods](docs/mathematical_methods.md), [protocol](docs/experimental_protocol.md), [references](docs/literature_notes.md), [troubleshooting](docs/troubleshooting.md).
# Interensic-Dimenalisty-
