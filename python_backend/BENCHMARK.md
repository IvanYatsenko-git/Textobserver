# Search benchmark

The test-corpus benchmark reads the Markdown, DOCX, and PDF files in `../test`. It removes near-duplicate documents, then uses the same paragraph grouping rules as `src/utils/segmentation.js`. Segment IDs therefore point back to the real test files. The current set contains 60 manually written queries (15 per unique document) plus the saved Gemini-generated queries.

## Run it

From this directory, with the project's Python environment activated:

```powershell
python test_corpus_benchmark.py
```

This reuses the saved Gemini query set and the hand-labeled set. It compares the production BGE-M3 + BM25 weighted fusion with semantic-only and BM25-only retrieval, plus legacy MiniLM and multilingual BGE reranker variants for reference. It can also compare multilingual E5. The reranker variants are loaded only by the benchmark; the application does not initialize or call them. By default, hybrid fusion considers 40 candidates, matching the app's usual top-5 search. Alternative models may need to download the first time and can take substantially longer on CPU.

To run only the hand-labeled cases, without Gemini:

```powershell
python test_corpus_benchmark.py --manual-only
```

To skip alternative model downloads and compare the current retrieval components:

```powershell
python test_corpus_benchmark.py --skip-alternatives
```

To see how a deeper candidate pool changes the ranking, pass the depth used by the app for larger result counts (up to 250):

```powershell
python test_corpus_benchmark.py --retrieval-depth 240
```

To generate a fresh set of Gemini queries, set `GEMINI_API_KEY` in the environment and run:

```powershell
python test_corpus_benchmark.py --regenerate-queries
```

Manual cases stay in `../test/retrieval_manual_queries.json` and are not overwritten when Gemini queries are regenerated. Each case can use one `expected_id`, or multiple `relevant_ids` when several passages answer the query.

## Add precise manual queries

Export the current segment IDs and text so labels can be selected from the actual corpus:

```powershell
python test_corpus_benchmark.py --export-corpus ../test/retrieval_benchmark_corpus.json
```

Add a case to `retrieval_manual_queries.json`:

```json
{
  "case_id": "manual-my-query-01",
  "query": "A realistic question phrased differently from the source text",
  "relevant_ids": ["Accessibility.md::seg_30", "Accessibility.md::seg_81"],
  "source": "Accessibility.md"
}
```

Prefer queries that need paraphrase, mix technical terms with natural language, use different languages, or ask for a specific fact. Label all passages that genuinely answer the query. The runner validates each ID against the current corpus and reports stale labels.

## Read the results

The console reports Hit@K, MRR, Recall@K, cluster-bootstrap intervals for Hit@5, and paired Hit@5 differences against the current production pipeline. The JSON report also includes direct paired comparisons between every model pipeline. Paired comparisons use the same queries for each model; queries targeting the same passage are resampled together to avoid overstating confidence from paraphrases of one passage. Hit@K tells whether at least one labeled passage appears in the first K results; MRR rewards higher placement; Recall accounts for queries with multiple correct passages. `../test/retrieval_benchmark_report.json` includes each query's gold passages, ranks, top ten results, previews, misses, benchmark-only reranker latency, and paired confidence intervals for every model pipeline.

Use the hand-labeled results as the primary comparison. Gemini-generated queries have one known target passage and are useful for broad coverage, but they are synthetic. Review individual misses and intervals; do not select a model based on a small aggregate-score difference alone.
