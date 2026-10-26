# Experimental protocol and preregistration checklist

This file/configuration is a preregistration-style specification, **not evidence of an externally registered study**. Freeze and timestamp it before examining evaluation results. Record verified model commit, exact dataset hash, tokenizer revision, seeds, primary split, layer indices, sample counts, inclusion rules, primary contrast and estimator, matching bin width, family resampling design, multiplicity rule and justified null assumptions.

Primary contrast: logical reasoning minus factual recall PR, raw centered covariance, final non-padding input token, each available transformer block separately. Embedding output is reported but is not a block. Default evaluation family split is six exploratory/two held-out families per category. BY is used for primary PR tests across selected layers. H2 task spectra, H3 layer profiles, H4 preprocessing/length/family/sample sensitivity and secondary metrics are separately labeled. No combined across-layer significance is invented.

1. Run mathematical tests and synthetic validation; inspect synthetic labels.
2. Audit source prompts, verify answers and review ambiguity. The shipped bank uses chemical recall and Python output prediction; expand for broader scientific claims.
3. Freeze an expanded dataset with **more independent families**, preferably new entity sets and reviewed short-answer formats. Choose inference units considering semantic/factual item dependence as well as template dependence. Default family bootstrap does not resolve crossed entity dependence.
4. Select GPU/CPU explicitly, extract Qwen once, preserve resolved revision and checkpoint metadata.
5. Examine exploratory output before finalizing choices, then freeze analysis configuration and run held-out evaluation exactly once. Configuration `split: evaluation` is primary; `split: exploratory` is exploratory; `all` is descriptive.
6. Compare equal-count subsets and length-bin-matched controls. Report failed matching and per-category sample/family counts. Increase dataset variety rather than adding padding to create artificial length matches.
7. Report category/layer estimates and all missing/poor-fit results. Family bootstrap uncertainty is conditional and not population-wide causal inference. Do not enable permutation p-values without a defensible exchangeability design.
8. Keep synthetic validation and real model results in different directories and ZIPs.

Pilot constraints: 200 prompts supply 50/category and 12 held-out prompts/category (two families). Centered covariance rank is at most 11 for that held-out pilot, far below the 1536-wide hidden state. A reasoning/facts ordering is neither guaranteed nor assumed. Increasing to 1000 prompts without adding families improves cloud sampling but not family-level independent replication.

Optional stronger design: manually curate multiple independently sampled semantic/template families per task with shared answer protocols and overlapping tokenizer-length strata; hold out entities and families, test multiple model/training seeds or revisions only when scientifically intended, and predefine a justified family permutation scheme. Task labels alone do not generate randomization. Model answer accuracy and causal interventions are separate projects.

## Superseding version-2 profile

The default configuration now uses the version-2 640-prompt bank described in `design_v2.md`; the earlier 200-prompt protocol above describes the preserved legacy pilot only. Evaluation has 80 prompts/10 named families per category; verified all-category token-length matching retains 52/category, including 10 factual and eight reasoning families. 500 raw-mode conditional family-bootstrap replicates are used. Constant-token embeddings are excluded from primary block inference. Preflight is enforced before weights load. No task permutation test is enabled without a justified exchangeability design. Human review and the independent-family assumption remain unresolved limitations; do not call nominal family labels proof of independence.
