# Troubleshooting

- Missing numeric dependencies: install `requirements.txt` in a clean Python 3.10–3.12 virtual environment. Install the package editable to make CLI imports work.
- Missing Torch/Transformers for extraction: `python -m pip install -e '.[model]'`. The synthetic path imports neither Transformers nor a model.
- CUDA requested but unavailable: choose a GPU in Colab or `--device cpu`. The actual execution device is recorded; CPU Qwen is slow and memory intensive.
- CUDA OOM: restart with batch size 1 using the same output folder. A compatible completed-batch checkpoint resumes. Eager attention and hidden states have transient memory costs; short prompts help. Float32 is now the T4 default; native bfloat16 is used only when supported. Explicit float16 can overflow; it is not auto-selected.
- Torchvision import/version conflicts with Transformers in some Colab images: this text-only pipeline does not require torchvision. Use a compatible environment; if removing an unused conflicting package, restart the runtime and rerun setup. Do not repeatedly downgrade PyTorch without preserving CUDA compatibility.
- Model access/download failure: verify network/model ID/cache permissions. Only the public Qwen2.5-1.5B model is default. No weights are supplied in the ZIP.
- Cache mismatch: preserve the old cache; select a fresh `--output-dir` for changed model revision/data/seeds/layers/device/dtype. Batch size alone may change when resuming. Avoid simultaneous extraction jobs writing one cache.
- Truncation: never automatic. Increase `max_tokens` after inspecting lengths, or curate prompts; exceeding the limit fails.
- Incomplete cache: rerun extraction; analysis refuses partially written arrays.
- No length overlap: report `no_common_length_bins`, inspect token distributions, refine dataset design. Do not substitute character count for Qwen tokens.
- Undefined TWO-NN: inspect duplicate/zero/equidistant points, sample count and CDF diagnostics. Bootstrap duplicates are the reason its interval is intentionally missing.
- Missing p-values: expected unless exchangeability is justified. Two held-out families are insufficient for the configured minimum bootstrap families. Increasing resampling count cannot fix independent-unit scarcity.
- Long analysis time: full layer-wise float64 SVD and family resampling run on CPU. Start with a subset of layers and fewer bootstrap replicates for exploration; use >=1000 resamples and enough independent families for stable publication intervals. Low-count smoke settings are validation only.
- ZIP summaries cannot rerun analysis without NPY arrays: copy cache separately or pass export `--include-activations`. Weights are always excluded from project export.
- Matplotlib/pyparsing deprecation warnings in an older preinstalled host: these were observed during local validation; use the pinned reference environment. They do not affect the numerical test outcomes.

- Nonfinite hidden representation: do not zero-fill or silently drop prompts. The old generic diagnostic did not distinguish shape errors from NaN/Inf. New diagnostics show layer, prompt IDs and counts. On T4 retry the whole extraction with `dtype: float32`, batch size 1 and a **fresh** output directory. Precision changes must not mix rows in the old cache. GPU BF16 is selected only when natively supported. FP32 uses more memory, so begin with one prompt per batch.
- Colab/IPython secondary traceback after SystemExit: the programmatic CLI now preserves the original exception; notebook extraction calls `extract(config)` directly. The extra inspect traceback was not the model failure.
