"""Evidence-grounded claim verification: retrieve passages, run NLI, aggregate a verdict."""
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).parent
CORPUS = ROOT / os.getenv("CORPUS", "data/corpus.jsonl")
RETRIEVER = os.getenv("RETRIEVER", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
NLI = os.getenv("NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
TOP_K = int(os.getenv("TOP_K", 5))
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", 0.35))  # below this a passage counts as irrelevant
DECISION = float(os.getenv("DECISION", 0.6))  # NLI prob needed to call SUPPORTED/REFUTED
MAX_CHARS = int(os.getenv("MAX_CLAIM_CHARS", 500))
EVALUATED_LANGS = {"en", "hi", "es"}  # languages with a measured test set; others are flagged

log = logging.getLogger("verify")
logging.basicConfig(level=logging.INFO)

corpus = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
retriever = SentenceTransformer(RETRIEVER)
# ponytail: brute-force cosine over an in-memory tensor; swap for FAISS once the corpus is >1M passages
emb_cache = CORPUS.with_suffix(".emb.pt")  # rebuilt whenever the corpus file is newer
if emb_cache.exists() and emb_cache.stat().st_mtime > CORPUS.stat().st_mtime:
    corpus_emb = torch.load(emb_cache)
else:
    corpus_emb = retriever.encode([p["text"] for p in corpus], convert_to_tensor=True, normalize_embeddings=True,
                                  batch_size=128, show_progress_bar=len(corpus) > 1000)
    torch.save(corpus_emb, emb_cache)
tok = AutoTokenizer.from_pretrained(NLI)
nli = AutoModelForSequenceClassification.from_pretrained(NLI).eval()
NLI_LABELS = [nli.config.id2label[i].lower() for i in range(nli.config.num_labels)]


def retrieve(claim: str, k: int = TOP_K):
    q = retriever.encode(claim, convert_to_tensor=True, normalize_embeddings=True)
    return [(corpus[h["corpus_id"]], h["score"]) for h in util.semantic_search(q, corpus_emb, top_k=k)[0]]


@torch.no_grad()
def entail(premises: list[str], claim: str) -> list[dict]:
    enc = tok(premises, [claim] * len(premises), truncation=True, padding=True, return_tensors="pt")
    probs = nli(**enc).logits.softmax(-1).tolist()
    return [dict(zip(NLI_LABELS, p)) for p in probs]


def verify(claim: str, language: str | None = None) -> dict:
    t0 = time.perf_counter()
    hits = [(p, s) for p, s in retrieve(claim) if s >= MIN_RELEVANCE]
    t1 = time.perf_counter()
    rels = entail([p["text"] for p, _ in hits], claim) if hits else []
    t2 = time.perf_counter()

    evidence = []
    for (p, score), r in zip(hits, rels):
        relation = max(r, key=r.get)
        evidence.append({
            "id": p["id"], "text": p["text"], "source_title": p["source_title"],
            "source_url": p.get("source_url"), "provenance": p.get("provenance"),
            "relevance_score": round(score, 3), "relation": relation, "relation_score": round(r[relation], 3),
        })

    ent = max((r["entailment"] for r in rels), default=0.0)
    con = max((r["contradiction"] for r in rels), default=0.0)
    warning = None
    if not hits:
        verdict, conf = "NOT_ENOUGH_EVIDENCE", None
        message = "No sufficiently relevant passage was found in the reference corpus, so the claim could not be checked."
    elif ent >= DECISION and con >= DECISION:
        verdict, conf = "NOT_ENOUGH_EVIDENCE", None
        message = "Retrieved passages conflict: some appear to support the claim and others to contradict it."
        warning = "Conflicting evidence — review the passages yourself."
    elif ent >= DECISION or con >= DECISION:
        verdict, conf = ("SUPPORTED", ent) if ent >= con else ("REFUTED", con)
        best = next(e for e in evidence if e["relation"] == ("entailment" if verdict == "SUPPORTED" else "contradiction"))
        message = f"“{best['source_title']}” {'supports' if verdict == 'SUPPORTED' else 'contradicts'} the claim (passage {evidence.index(best) + 1})."
    else:
        verdict, conf = "NOT_ENOUGH_EVIDENCE", None
        message = "Related passages were found, but none clearly supports or contradicts the claim."

    if language and language not in EVALUATED_LANGS:
        warning = (warning + " " if warning else "") + f"Language '{language}' has not been evaluated; results may be less reliable."

    return {
        "claim": claim, "language": language, "verdict": verdict,
        "confidence": round(conf, 3) if conf is not None else None,
        "evidence": evidence, "message": message, "warning": warning,
        "timings_ms": {"retrieval": round((t1 - t0) * 1000), "nli": round((t2 - t1) * 1000)},
    }


app = FastAPI(title="Claim Verifier")
# ponytail: in-memory aggregate stats, no claim text kept; lost on restart — move to a DB if they need to persist
stats = {"requests": 0, "errors": 0, "verdicts": Counter(), "latency_ms": []}


class VerifyRequest(BaseModel):
    claim: str = Field(min_length=1, max_length=MAX_CHARS)
    language: str | None = Field(default=None, max_length=10)


@app.exception_handler(RequestValidationError)
async def bad_request(_: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": exc.errors()[0]["msg"]})


@app.exception_handler(Exception)
async def server_error(_: Request, exc: Exception):
    stats["errors"] += 1
    log.exception("verification failed")
    return JSONResponse(status_code=500, content={"error": "Verification failed due to a server error. Please try again."})


@app.post("/api/v1/verify")
def verify_endpoint(req: VerifyRequest):
    claim = " ".join(req.claim.split())  # collapse whitespace only; meaning untouched
    if not claim:
        return JSONResponse(status_code=422, content={"error": "Claim must not be empty."})
    t = time.perf_counter()
    result = verify(claim, (req.language or None) and req.language.lower())
    ms = (time.perf_counter() - t) * 1000
    stats["requests"] += 1
    stats["verdicts"][result["verdict"]] += 1
    stats["latency_ms"] = (stats["latency_ms"] + [ms])[-1000:]
    log.info("verdict=%s chars=%d ms=%.0f", result["verdict"], len(claim), ms)
    return result


@app.get("/api/v1/stats")
def get_stats():
    lat = sorted(stats["latency_ms"])
    return {
        "requests": stats["requests"], "errors": stats["errors"], "verdicts": stats["verdicts"],
        "p95_latency_ms": round(lat[int(0.95 * (len(lat) - 1))]) if lat else None,
        "corpus_passages": len(corpus), "models": {"retriever": RETRIEVER, "nli": NLI},
    }


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")
