# Corrected version-2 Colab upload guide

Download **`Qwen_Colab_V2.ipynb`** and **`intrinsic-dimension-transformers-v2-colab.zip`** from the project root. Preserve the completed 200-prompt pilot and its results separately.

1. Open https://colab.research.google.com/ and choose **File → Upload notebook**; upload the v2 `.ipynb`.
2. Select **Runtime → Change runtime type → GPU**, choosing T4 if available. Save/connect. GPU allocation varies; real extraction stops if CUDA is unavailable.
3. Run the first code cell. Upload the v2 ZIP when prompted; do not unzip it yourself. The first cell checks the exact source fingerprint and requests the new ZIP if the old pilot source is still present. If old Python modules were already loaded, it asks for a session restart after updating files; cached weights and files can then be reused.
4. Run installation. Compatible installed packages are retained. If a preloaded version changes, restart the session and rerun setup.
5. Run the Drive cell and authorize access. Drive is enabled at your request for this run. Results/checkpoints go to **`MyDrive/intrinsic_dimension/qwen_v2_fp32`**. The old pilot folder is preserved.
6. Run configuration/tests. The model is pinned to the pilot commit; extraction is explicitly **CUDA float32, batch size 1**. All 29 hidden-state outputs are available; embedding is a diagnostic rather than a primary block comparison.
7. Run the dataset/preflight cell **before extraction**. Expected evaluation counts: 80 prompts/10 named families per category. Expected all-category length-matched counts: 52/category, retaining factual/reasoning families 10/8. The notebook recomputes actual tokenizer counts; failure stops before model weights load. Family labels do not establish independent sampling; human review is pending.
8. Run extraction. This needs **new activations for all 640 prompts**; the old 200-row cache cannot be reused. Cached model weights may be reused in the same runtime/cache. No training is performed. If CUDA runs out of memory, keep batch size 1 and review actual free GPU memory; do not switch dtype into the same partially populated cache.
9. Run analysis/report. Raw conditional spectral intervals use 500 family-bootstrap replicates. Unit/standardized modes are descriptive sensitivity checks. Length controls include matching, equal-count control and equal-family-composition control. CPU analysis may take time after GPU extraction.
10. Run export. Download **`intrinsic_dimension_v2_results.zip`**; a copy is also saved in Drive. It contains code, configs, metadata, tables, reports and all HD figures. Activations stay in the Drive run directory; optionally include them with `INCLUDE_ACTIVATIONS=True`. Model weights are excluded.

All seven figure families export as **600-DPI PNG plus vector PDF/SVG**, with larger typography and source CSVs. Notebook previews are scaled; use downloaded PDF/SVG for publication or zooming. Raster dimensions vary by layout (several thousand pixels across). `figures/manifest.json` records export quality.

Inside the results ZIP, figures are under `intrinsic-dimension-transformers/artifacts/exported_run/figures/`, tables/report under `.../results/`. The upload ZIP contains source only, without model outputs or fabricated scientific figures.

Reports distinguish **Not performed** (no justified significance-testing design), **Not estimable** (e.g. zero variance/invalid neighborhoods) and conditional intervals. No p-value is manufactured simply to eliminate a missing field. Repeated constant final-token embeddings are expected to have undefined covariance dimensionality; transformer block activations remain contextual and informative when variance is nonzero.

The exact-tokenizer preflight, data/schema/ground-truth logic, mathematical tests, synthetic v2 pipeline and local notebook validation were checked. Interactive Drive/upload and actual v2 CUDA Qwen extraction require verification in your Colab run. See `design_v2.md` for the assumptions and remaining limitations.

Official Colab import/runtime/resource information: https://research.google.com/colaboratory/faq.html
