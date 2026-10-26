# Validation performed in this workspace

Date: 2026-10-11. Actual execution: CPU on macOS ARM64. No CUDA GPU or full Qwen weights were used. Model dependencies Transformers/tokenizers/huggingface-hub were absent; the friendly dependency error was exercised. No real-model scientific conclusion is available.

- `PYTHONPATH=src python3 -m pytest -q --disable-warnings`: **16 passed**. Mathematical formulas, scaling invariance, zero/negative spectra, covariance consistency, centering invariance, rank tolerance, known PCA thresholds, approximate TWO-NN Gaussian dimensions (2 and 5), duplicate handling, schema/duplicate checks, reproducible generation/sampling, controls, FDR, family bootstrap/permutation, attention-mask indexing, layer mapping, synthetic end-to-end output and ZIP contents.
- Real PyTorch inference-mode extraction path was exercised using a local deterministic fake tokenizer/backbone. Checked both padding conventions in mathematical selector tests; fake backbone uses left padding. Verified interruption after a completed batch, compatible resume with changed batch size, cache mismatch rejection, checkpoint completeness, sample order, layer outputs and no gradient tracking. This is not a full Hugging Face integration test.
- Default CLI dataset generation/audit and 200-prompt schema validation succeeded. A separate 1000-prompt generation succeeded. CPU-small 32-prompt dataset was generated.
- `PYTHONPATH=src python3 -m intrinsic_dim.cli smoke-test --config configs/smoke_test.yaml`: succeeded; seeded synthetic arrays, all estimators, source CSVs, seven PNG/PDF figure sets, comparison/control diagnostics and reports produced. No length overlap exists in this synthetic proxy-token example; matching failure is recorded honestly.
- All six notebooks passed nbformat validation and code-cell syntax compilation. All six then executed via **nbclient with local Jupyter kernels**, preserving executed notebooks in `artifacts/notebook_validation/`. Model/Drive/upload/install/download toggles were false. The initial sandbox prevented Jupyter loopback ports; execution succeeded with the approved local-kernel command.
- Compilation/import checks and CLI help succeeded. Extraction correctly failed with a helpful message when optional model dependencies were absent.
- Generated figures were visually inspected, including all required plot types. Synthetic labeling, readable legends and control failure labels were checked.
- ZIP archive created and checked for CRC integrity, expected source/notebook/report content and exclusion of activation NPY arrays and weights.

Local versions: Python 3.10.9; NumPy 1.26.4; SciPy 1.15.3; Matplotlib distribution metadata 3.8.0; PyYAML 6.0.3; PyTorch 2.11.0. The host produced Matplotlib/pyparsing deprecation warnings (257 in the suite), not numerical failures. The package version metadata does not guarantee a clean preinstalled environment. `requirements.txt` defines a fresh pinned numerical target; installing that fresh target was not executed here. Optional model versions were not installed.

Untested: full Qwen download/forward on HF Transformers 4.46.3, CUDA/T4 runtime and peak VRAM, interactive Colab upload/Drive mounting/download, fresh dependency installation. Optional Levina–Bickel, calibrated TWO-NN goodness-of-fit p-values, crossed entity/template bootstrap and causal interventions are not implemented. Natural-template permutation exchangeability is not assumed; inference remains unavailable where the design does not support it.

## Colab upload and HD update

The ready notebook is `Qwen_Colab_Ready.ipynb` (same source as `notebooks/colab_pipeline.ipynb`). Both pass nbformat/schema and cell compilation. The ready notebook executed successfully via nbclient in local-validation mode: interactive upload, dependency installation, Drive, Qwen/CUDA and Colab download paths were skipped. Its synthetic pipeline, audit, report, previews and results export ran successfully. The suite remains **16 passed**; the updated renderer produced 337 host deprecation warnings, not test failures.

All seven actual synthetic figure outputs were regenerated and visually inspected. PNG metadata reports approximately 600 DPI; dimensions are 7023×4669, 8224×5844, 9422×3198, 5823×4080, 9423×3492, 8824×3492 and 5824×4080 pixels. Corresponding PDF and SVG vector files plus source CSVs exist. Tests use a lower explicit render DPI to keep repeated validation lightweight; project and notebook defaults are 600 DPI.

The new source-only upload ZIP is `intrinsic-dimension-transformers-colab.zip`. It contains the project folder, updated configuration/notebooks and 200-prompt dataset; no model weights, activation caches or scientific outputs are bundled. Results are generated in the user's actual Colab runtime and downloaded separately.

## Precision failure recovery

A user reported successful CUDA/model loading followed by the original generic shape/nonfinite rejection on the first FP16 batch in Colab Python 3.13. That diagnostic cannot by itself prove overflow or distinguish shape from nonfinite values. The patch makes these failures distinct, reports offending layer/prompt IDs/NaN/Inf counts, and never advances the checkpoint for an invalid batch. Default precision now selects native BF16 when supported and FP32 otherwise (including T4); explicit FP16 remains available with a warning. Changed precision requires a fresh cache. Notebook extraction uses the function directly, and programmatic CLI preserves original exceptions rather than producing SystemExit followed by IPython's secondary inspect failure.

**19 tests passed** after the patch, including native-BF16/FP32 policy, preservation of programmatic CLI exceptions, and failure without committing a nonfinite batch. The updated notebook passes schema/cell compilation checks. The actual Colab CUDA failure was not reproduced locally, and FP32 recovery requires confirmation from the user's GPU run. `scripts/colab_recover_fp32.py` can be pasted into the existing notebook after setup/configuration cells; it uses the existing weight cache and a separate `_fp32` output directory.

Precision sources: Qwen model config https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/config.json and PyTorch numerical accuracy https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html.

## Version-2 design correction

The v2 bank contains 640 prompts (160/category), 20 named families/category, eight prompts/family and a fixed 10-family/80-prompt evaluation split. Family names are operational clusters and not proof of independent population sampling. Exact tokenizer measurements from the pilot commit `8faed761d45a263340a0528343f099c05c9a4323` were performed using a temporary Transformers 4.46.3/tokenizer environment and tokenizer-only files. **No model weights were downloaded locally.**

Preflight passed: all-category evaluation length matching retains **52 prompts/category**, across four-token bins 40–43 and 44–47. Factual/reasoning matched family counts are **10/8** (arithmetic 8, coding 7). Stored preflight fingerprints and counts accompany the dataset. Colab recomputes tokenizer counts before model load. Duplicate/schema checks passed; controlled Python expressions were evaluated against the generated reference answers; seat/color constraints were independently enumerated. Source-linked factual fixtures remain subject to human ambiguity review.

**25 tests passed** after the redesign (host Matplotlib/pyparsing warnings, not numeric failures). Additional tests cover v2 deterministic generation/family splits, actual tokenized preflight versus legacy failure, generated-code ground truth, Gram/SVD bootstrap equivalence, real conditional-interval availability in synthetic v2 end-to-end analysis, and equal-count/equal-family length controls. The v2 synthetic end-to-end test additionally exercised N<d with 128-wide synthetic arrays; six design tests passed in that targeted run.

`Qwen_Colab_V2.ipynb` and the updated canonical notebooks passed schema/cell compilation. The v2 ready notebook executed through nbclient in local-validation mode: configuration, bundled verified-token preflight, tests, synthetic analysis/report, previews and export succeeded. Interactive upload, installation, Drive authorization, CUDA and actual v2 Qwen model extraction were skipped locally. Drive defaults to enabled in Colab at the user's explicit request and is disabled in local validation.

`smoke-test --config configs/design_validation.yaml` runs all v2 sampling/controls on synthetic activations, including 500 raw-mode family-bootstrap replicates and high-resolution PNG/PDF/SVG exports. Synthetic summaries remain marked as validation, never model findings. The corrected report distinguishes expected constant-token embeddings, unperformed significance tests, conditional intervals and diagnostic flags. No exchangeability claim or p-value was manufactured to fill a missing field.

The source-only v2 ZIP was built and checked for integrity, required notebook/config/data/module content and exclusion of weights/activations/scientific output. Actual v2 GPU results await the user's new Colab run. Preserve the 200-prompt run as the legacy pilot; do not reuse its arrays for the new 640-prompt bank.

Final freeze checks: the live pinned tokenizer preflight passed again through the actual `prepare_dataset` function after authored-input fingerprint enforcement. An input-tampering test confirmed failure before tokenizer download. The ready notebook source-version guard passed locally, preventing the stale 200-prompt source from silently being reused.

Final full suite: **26 passed** after the frozen-input gate, source-version guard and input-tampering regression test. Host renderer deprecation warnings did not affect numerical or data checks.
