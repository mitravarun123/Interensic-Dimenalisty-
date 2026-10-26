# Prompt data

`prompts.jsonl` contains 200 deterministic generated prompts (seed 42), 50 in each category. Eight explicit template families per category: families 0–5 exploratory, 6–7 evaluation. Split occurs at family level, not row level. Exact normalized duplicates are rejected; token Jaccard near duplicates, reused entities, counts, answer formats and lengths are audited. Equal counts are not proof of matching. Semantic duplicates and repeated entities remain.

Required JSONL fields: `prompt_id`, `task_category` (facts/arithmetic/reasoning/coding), `template_id`, `prompt`, `expected_answer` (nullable), `token_count` (positive integer or null), `token_count_method`, `split` (exploratory/evaluation), `generation_seed`, `entities` (list), `answer_format`. Optional fields include `template_family_index`, `answer_verification`, `reviewer`, `review_date`, `review_status`.

Ground truth: first 50 IUPAC element names/symbols/atomic numbers (https://iupac.org/what-we-do/periodic-table-of-elements/); integer answers computed in Python; logic answers checked from Boolean truth rules and strict relation order; coding answers computed directly with the requested built-in semantics, never through arbitrary code execution. No LLM answer is used as truth.

`manually_reviewed.example.jsonl` is a schema example **awaiting human review**, not a claim of completed review. Copy to your own file, review ground truth and ambiguity, record reviewer/date and select with `--dataset-path`. No dataset scraping or paid service is needed. The factual template bank supports 400 distinct prompts; a balanced total of 1000 works, but expansion beyond the bank capacity fails explicitly. More rows alone do not create more independent template families.

The default prompts initially have `token_count=null`. Extraction saves a separate tokenized dataset with exact model-token counts. The original source remains unchanged. Length matching uses bins of four tokens by default (configurable); no common bin is reported explicitly. One shared final answer-only suffix reduces but does not eliminate answer-format confounds.

## Version 2 (now the default)

`prompts_v2.jsonl`: 640 prompts, 20 named families per category, eight prompts per family; even family indices exploratory and odd evaluation. Each category has 160 total prompts and 80 held-out prompts in 10 families. `factual_fixtures_v2.json` contains fixed authoritative reference triples and links. Computed arithmetic, enumerated constraints/Boolean rules and controlled Python expressions verify other ground truth; human ambiguity review remains pending.

Exact tokenizer lengths for pilot commit `8faed761d45a263340a0528343f099c05c9a4323` were verified locally; `preflight_v2_verified.json` records them and the source fingerprint. Colab recomputes lengths before model weights load. Matching retains 52 evaluation prompts/category with factual/reasoning family counts 10/8. See `docs/design_v2.md` for design assumptions and remaining confounds. Old `prompts.jsonl` and eight-family generation are retained for the legacy pilot/synthetic tests and are not relabeled as new independent families.
