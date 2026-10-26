# Revised design after the 200-prompt pilot

The legacy run completed successfully but its default evaluation split had only 12 prompts/two families per category and no common model-token length bin. That limited inference; it was not an activation-extraction failure. Preserve that run as the pilot. The new run requires new activations; old 200-row arrays cannot represent this dataset.

## Version-2 specification

- 640 prompts, 160 per category.
- 20 named task families per category, eight distinct prompts each.
- Fixed family-level split: even family indices exploratory, odd indices evaluation. Each category has 10 held-out families/80 evaluation prompts. IDs, design version and source references are recorded.
- Primary model pinned to the exact pilot commit `8faed761d45a263340a0528343f099c05c9a4323`. CUDA FP32, batch size 1. Fresh output/cache for activations: `qwen_v2_fp32`; the model weight download cache can be reused.
- Same final non-padding input-token convention and common answer-only suffix. No irrelevant padding or random final-token markers are used to create spurious variance. Constant final-token embeddings remain noninformative diagnostics and are excluded from primary block inference.
- Fixed/source-linked recall spans chemistry, country/language identifiers, SI units/prefixes, constellation names, planetary classification, postal and spelling conventions. Arithmetic spans 20 operations/compositions. Logic spans explicit order, class/implication, Boolean, constraint, graph and set problems. Coding spans 20 verified Python literal operations. These operationalizations remain narrower than general reasoning/coding ability.
- Natural wording makes prompt-length distributions overlap. Matching uses the **actual Qwen tokenizer**, never word-count proxies. The question instructions clarify the requested notation/arithmetic without inserting nonsense padding. Different instructions/syntax remain residual confounds.

## Verified preflight

The actual tokenizer from the pinned model commit was loaded locally; no weights were downloaded. The shipped `prompts_v2.jsonl` carries these exact token counts. `preflight_v2_verified.json` contains the fingerprint and counts. The notebook recomputes them using its real tokenizer before weights load, so stale counts cannot pass unchecked.

Evaluation length matching in bins of four tokens retains **52 prompts per category**, across bins 40–43 and 44–47. Retained family counts: facts 10, arithmetic 8, reasoning 8, coding 7. Primary raw evaluation estimates use 80 prompts/10 families per category. The length-control CI uses 52 factual/52 reasoning prompts with 10/8 observed families. These are empirical design checks, not findings about representation dimensionality.

Preflight requires at least 64 evaluation prompts/eight families per category, at least 32 four-category length-matched prompts, and at least eight matched factual and reasoning families. Failure stops extraction before loading model weights. Matched-family coverage differs across categories; this is reported, not called perfect matching. Within-bin token differences remain. Exploratory all-category length matching is weaker and is not the primary confirmatory control.

## Inference and scientific limits

500 family-bootstrap replicates produce conditional percentile intervals for raw spectral summaries. Primary analysis remains evaluation reasoning minus facts PR across transformer blocks. Unit normalization and standardized-feature analyses are separately labeled descriptive controls; their resampling is disabled by default to reduce unnecessary CPU work. Duplicate primary/evaluation subsets reuse computation. Equivalent small Gram covariance spectra accelerate bootstrap calculations; primary point estimates/rank diagnostics still use economy SVD. Gram/SVD agreement is tested.

A named family is a meaningful template/semantic block, **not proof of an independently sampled experimental unit**. Several tasks share abstract operators, sources, entities or vocabulary. Bootstrap intervals assume independently sampled family clusters and condition on the chosen empirical family population. The curated bank is not a random sample from all reasoning tasks. Crossed entity/operator dependence and human ambiguity review remain limitations. Ten families is a better exploratory basis than two, not an assurance of calibrated publication-level inference.

No p-value is created by switching on arbitrary task-label permutations. Such a test requires a scientifically justified exchangeability/randomization design. More templates do not provide that automatically. Reports label unperformed tests distinctly from undefined metrics and show interval directions without claiming automatic significance or support for H1. TWO-NN replacement-bootstrap remains invalid and disabled; local-fit failures remain visible.

The version-2 dataset was designed after examining the pilot report. Freeze it and its configuration before the next extraction, preserve the pilot separately, document deviations, and treat replication or a genuinely preregistered study as further validation. Human review is pending. This improves the pilot design without pretending that every scientific limitation is eliminated.

## Run

Upload the new notebook and `intrinsic-dimension-transformers-v2-colab.zip` in Colab. Drive storage is enabled at the user's request; authorize the mount. Run cells top to bottom, including preflight. Checkpoints and results go to `MyDrive/intrinsic_dimension/qwen_v2_fp32`. All seven figure families retain 600-DPI PNG and vector PDF/SVG exports. The last cell downloads `intrinsic_dimension_v2_results.zip` and also saves it in Drive.

The missing HF_TOKEN warning is optional for this public model. GPU extraction was not executed locally for version 2. Exact tokenizer lengths, data/ground-truth logic, estimators, synthetic analysis, preflight and packaging are validated separately.

Length-control contrasts also include `length_count_control` (same sample count, no length filtering) and `length_family_control` (same count within each retained family, no length filter). This prevents confusing the 80-versus-52 sample-size change with a length effect, and exposes effects of changed family composition. These are observational sensitivity checks, not random task-label assignment or perfect confound removal.
