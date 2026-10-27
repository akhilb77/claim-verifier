# Claim Verifier

Evidence-grounded, cross-lingual claim verification. Paste a claim, get **Supported / Refuted / Not Enough Evidence**, the evidence passages behind it, and their sources.

**Pipeline:** claim → multilingual semantic retrieval (`paraphrase-multilingual-MiniLM-L12-v2`, top-5 above a relevance cut-off) → multilingual NLI on each passage (`mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`) → verdict (≥0.6 entailment or contradiction, else NEE; conflicting evidence → NEE with a warning).

Verdicts are model predictions over a limited corpus, not proof of truth.

## Run

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app:app --port 8000          # demo corpus, open http://localhost:8000
python eval.py                       # demo test set (en / hi / es)
```

API: `POST /api/v1/verify` `{"claim": "...", "language": "en"}` · aggregate stats: `GET /api/v1/stats`.

Config via env vars: `CORPUS`, `TOP_K`, `MIN_RELEVANCE`, `DECISION`, `MAX_CLAIM_CHARS`, `RETRIEVER`, `NLI_MODEL`.

## FEVER

```bash
mkdir -p data/fever && cd data/fever
curl -LO https://fever.ai/download/fever/train.jsonl
curl -LO https://fever.ai/download/fever/paper_dev.jsonl
curl -LO https://fever.ai/download/fever/wiki-pages.zip   # 1.7 GB
cd ../..
python fever_prep.py 1000                                  # corpus + 1000-claim dev sample
head -n 200 data/fever/dev_sample.jsonl > data/fever/dev_200.jsonl
python baseline.py data/fever/dev_200.jsonl                # TF-IDF + LogisticRegression
CORPUS=data/fever/corpus.jsonl python eval.py data/fever/dev_200.jsonl
CORPUS=data/fever/corpus.jsonl uvicorn app:app --port 8001
```

The FEVER corpus is **not** the full 5.4M-page dump: it is every page cited as evidence by the dev sample plus ~0.1% random distractor pages (30,638 sentences, 5,969 pages). Retrieval is therefore easier than open-domain FEVER.

## Results

| System | Test set | n | Macro-F1 | Accuracy | Recall@5 |
|---|---|---|---|---|---|
| TF-IDF + LR (claim only) | FEVER paper_dev (full) | 9,999 | 0.569 | 0.568 | n/a |
| TF-IDF + LR (claim only) | FEVER dev_200 | 200 | 0.504 | 0.500 | n/a |
| Retrieval + NLI pipeline | FEVER dev_200 | 200 | *running* | | |
| Retrieval + NLI pipeline | Demo, overall | 27 | 0.825 | 0.852 | 1.0 |
| Retrieval + NLI pipeline | Demo, English | 15 | 0.855 | 0.867 | 1.0 |
| Retrieval + NLI pipeline | Demo, Hindi → English evidence | 6 | 1.000 | 1.000 | 1.0 |
| Retrieval + NLI pipeline | Demo, Spanish → English evidence | 6 | 0.667 | 0.667 | 1.0 |

- **Baseline** sees only the claim text (no evidence), trained on all 145,449 FEVER train claims. Its above-chance score comes from claim-wording bias, so it is a floor, not a verifier.
- **Grounded accuracy** counts a verdict as correct only if a complete gold evidence set was shown (FEVER-score style).
- Models are used pretrained (no FEVER fine-tuning). Thresholds were not tuned on these test sets.
- The demo set (27 claims, hand-written, en/hi/es) is too small for strong conclusions; it shows cross-lingual retrieval works with English-only evidence.

## Files

| File | Purpose |
|---|---|
| `app.py` | FastAPI backend: retrieval, NLI, verdict logic, API |
| `index.html` | Web UI |
| `eval.py` | Pipeline evaluation (macro-F1 per language, Recall@k, MRR, grounded accuracy) |
| `baseline.py` | TF-IDF + LogisticRegression baseline |
| `fever_prep.py` | Builds FEVER corpus and dev sample |
| `data/corpus.jsonl`, `data/test.jsonl` | Demo corpus and multilingual test set |
| `results/` | Saved evaluation JSON |
