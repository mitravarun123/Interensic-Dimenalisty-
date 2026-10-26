# Representation geometry research report

**Measured model activations; observational geometry study.**

## Question and hypotheses
Do logical reasoning prompts exhibit higher participation ratio than factual recall prompts? H1 is directional; H2 tests task differences, H3 layer variation, and H4 persistence under controls. No ordering is assumed.

## Model and extraction
Model: `Qwen/Qwen2.5-1.5B`; resolved revision: `8faed761d45a263340a0528343f099c05c9a4323`; actual device: `cuda`; dtype: `float32`.
Final non-padding input token; prompt only, no generation; standard HF hidden_states tuple, final output after RMSNorm.
Hidden-state tuple index 0 is embedding output for real extraction; indices 1..L map to blocks. For Qwen the last returned vector includes final RMSNorm. Synthetic layers have no transformer interpretation.

## Dataset and controls
Rows: 200; selected split: evaluation. Category counts: {'facts': 50, 'arithmetic': 50, 'reasoning': 50, 'coding': 50}
Explicit templates use fixed chemical fixtures, computed arithmetic, unambiguous Boolean/relational logic, and verified Python output semantics. All prompts request short answers. Token counts are exact tokenizer counts in model extraction; synthetic counts are whitespace proxies.
Length matching: {'status': 'no_common_length_bins', 'bin_width': 4, 'shared_bins': [], 'matched_count': 0, 'warning': 'Bin matching leaves within-bin residual length differences.'}. Common-count random subsampling is used per layer. Template evaluation families are never used to fit the 2D PCA projection.
Entity reuse, shared vocabulary, residual bin lengths, answer types and semantic diversity remain confounds; equal counts do not imply matched categories.

## Mathematical methods
Centered sample covariance eigenvalues are σ²/(N−1), from float64 economy SVD. At most min(N−1,d) nonzero centered directions can be identified. PR=(Σλ)²/Σλ²; covariance effective rank=exp(−Σp log p), p=λ/Σλ. PCA thresholds (80/90/95/99%) are separate summaries, not intrinsic-dimension ground truth. TWO-NN fits −log(1−F) on log(r₂/r₁) through the origin, dropping the largest 10% ratios. Duplicate clouds are invalid for TWO-NN; poor CDF fits are flagged.
Raw, unit-normalized and standardized modes have separate labels. Unit normalization changes the geometry; feature scaling changes covariance spectra.

## Estimates and primary comparisons
Complete per-category/per-layer estimator results: `estimates.csv`. Covariance and singular spectra: `spectra.csv`. Full estimator and distance diagnostics: `diagnostics.json`. All six task-pair differences and ratios (descriptive) are in `pairwise_comparisons.csv`.
| Layer | PR difference | Bootstrap interval | Adjusted p | Outcome |
|---|---:|---|---|---|
| Embedding output | None | [None, None] | None | inconclusive_inference_unavailable |
| Block 1 | 0.09368901356547421 | [None, None] | None | inconclusive_inference_unavailable |
| Block 2 | 0.15948169832207704 | [None, None] | None | inconclusive_inference_unavailable |
| Block 3 | 0.2747427525976691 | [None, None] | None | inconclusive_inference_unavailable |
| Block 4 | 0.3127848245563729 | [None, None] | None | inconclusive_inference_unavailable |
| Block 5 | 0.32213527296619526 | [None, None] | None | inconclusive_inference_unavailable |
| Block 6 | 0.3844862308705217 | [None, None] | None | inconclusive_inference_unavailable |
| Block 7 | 0.3469846629584903 | [None, None] | None | inconclusive_inference_unavailable |
| Block 8 | 0.26205129247196224 | [None, None] | None | inconclusive_inference_unavailable |
| Block 9 | 0.2405033133344745 | [None, None] | None | inconclusive_inference_unavailable |
| Block 10 | 0.21061204946381595 | [None, None] | None | inconclusive_inference_unavailable |
| Block 11 | 0.19951310218628815 | [None, None] | None | inconclusive_inference_unavailable |
| Block 12 | 0.24086890685551943 | [None, None] | None | inconclusive_inference_unavailable |
| Block 13 | 0.19419550993114898 | [None, None] | None | inconclusive_inference_unavailable |
| Block 14 | 0.15454174930382147 | [None, None] | None | inconclusive_inference_unavailable |
| Block 15 | 0.13636968109931225 | [None, None] | None | inconclusive_inference_unavailable |
| Block 16 | 0.11939841859870226 | [None, None] | None | inconclusive_inference_unavailable |
| Block 17 | 0.13525295975846263 | [None, None] | None | inconclusive_inference_unavailable |
| Block 18 | 0.08020771250611092 | [None, None] | None | inconclusive_inference_unavailable |
| Block 19 | 0.040905409392978154 | [None, None] | None | inconclusive_inference_unavailable |
| Block 20 | 0.06428390875477952 | [None, None] | None | inconclusive_inference_unavailable |
| Block 21 | -0.13804045878657045 | [None, None] | None | inconclusive_inference_unavailable |
| Block 22 | -0.23556795318110924 | [None, None] | None | inconclusive_inference_unavailable |
| Block 23 | -0.7549030006179869 | [None, None] | None | inconclusive_inference_unavailable |
| Block 24 | -1.472055249044458 | [None, None] | None | inconclusive_inference_unavailable |
| Block 25 | -1.139546220472657 | [None, None] | None | inconclusive_inference_unavailable |
| Block 26 | -1.0460418285050455 | [None, None] | None | inconclusive_inference_unavailable |
| Block 27 | -0.8716169035544441 | [None, None] | None | inconclusive_inference_unavailable |
| Block 28 + final RMSNorm | -0.10950117719099373 | [None, None] | None | inconclusive_inference_unavailable |

## Uncertainty and testing
Template families are resampled as whole clusters. No per-prompt dimension observations are invented. Percentile bootstrap intervals condition on the template population and vary sample counts as clusters are resampled. They are unreliable with few families; the configured minimum is enforced. TWO-NN bootstrap intervals are disabled because repeated points invalidate nearest-neighbor distances.
No permutation p values are computed unless family exchangeability is explicitly asserted in configuration. Natural task-specific templates do not establish exchangeability; confidence intervals alone do not establish significance. Opt-in permutation tests require a stronger null than equal PR. BY correction handles dependence across layers and metrics; primary PR layer tests and secondary/robustness tests form separate families.
Configured bootstrap/permutation replicates: 200/200. Full valid-replicate counts, raw differences, ratios, flags and adjusted p values: `comparisons.csv`.

## Robustness and failures
Repeated common-count subsampling, separate preprocessing modes, length matching and exploratory versus held-out template analyses are stored in `robustness.csv`, `comparisons.csv` and `selection.json`. Figure 5 shading is a descriptive range, not a confidence interval.
Failed/unstable estimates: 256; see `results_summary.json` and `diagnostics.json`. Missing intervals/p values remain explicitly null.

## Limitations and interpretation
A 200-prompt pilot supplies only about 50 prompts per category and two held-out template families. It cannot establish a broad claim about reasoning. Expand independent template families, manually review prompts, preregister the exact model revision and analysis before examining held-out output. The shipped chemical recall and Python prediction tasks are narrow operationalizations. H2–H4 are descriptive robustness analyses; no confirmatory all-pairs category inference is claimed.
Dimensionality is one aspect of representation geometry. It does not establish causal mechanisms, computational effort, reasoning ability, representational content, answer correctness, or a complete mechanistic explanation. No answers are generated in the primary experiment. Higher width is not higher intrinsic dimensionality. Null tests do not demonstrate equality.
Synthetic validation establishes software behavior only. For real model findings, run extraction on Qwen and preserve resolved revision and metadata.

## Reproducibility
Analysis fingerprint: `d0e85adceeec469cde778b734e863203c2c35b8a903e24bf92cf67cab5aac003`. Source dataset IDs and selected subsets are preserved. Seeds and full configuration/software metadata appear in `analysis.json` and activation `metadata.json`.
Figures are exported to PNG/PDF and carry source CSVs. The ZIP export excludes activation arrays and model weights unless explicit cache inclusion is selected.
