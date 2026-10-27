"""Evaluate the pipeline: per-language macro-F1, retrieval Recall@k / MRR, evidence-grounded accuracy.

usage: python eval.py [test.jsonl]      (set CORPUS=... to match the test set's corpus)
gold is a passage id, a list of evidence sets (FEVER: each set is a list of sentence ids), or null for NEE.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import classification_report, f1_score

from app import TOP_K, retrieve, verify

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/test.jsonl")
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
by_lang = defaultdict(list)
t0 = time.perf_counter()

for i, r in enumerate(rows, 1):
    gold = [[r["gold"]]] if isinstance(r["gold"], str) else r["gold"]
    out = verify(r["claim"], r["language"])
    ranked = [p["id"] for p, _ in retrieve(r["claim"], TOP_K)]  # raw ranking, before relevance cut-off
    shown = {e["id"] for e in out["evidence"]}
    correct = out["verdict"] == r["label"]
    gold_ids = {g for s in gold for g in s} if gold else set()
    by_lang[r["language"]].append({
        "gold": r["label"], "pred": out["verdict"], "has_ev": bool(gold),
        "hit": bool(gold_ids & set(ranked)),  # any gold sentence in top-k
        "full_hit": bool(gold) and any(set(s) <= set(ranked) for s in gold),  # a complete evidence set in top-k
        "rr": next((1 / k for k, pid in enumerate(ranked, 1) if pid in gold_ids), 0.0),
        # PRD 10.2 / FEVER score: a correct verdict only counts if a full gold evidence set was shown
        "grounded": correct and (not gold or any(set(s) <= shown for s in gold)),
    })
    if not correct and len(rows) <= 50:
        print(f"  miss [{r['language']}] {r['claim'][:60]!r}: gold={r['label']} pred={out['verdict']}")
    if i % 100 == 0:
        print(f"  {i}/{len(rows)} ({(time.perf_counter() - t0) / i:.2f}s/claim)", flush=True)


def summary(res):
    ev = [x for x in res if x["has_ev"]]
    rate = lambda key, xs: round(sum(x[key] for x in xs) / len(xs), 3) if xs else None
    return {
        "n": len(res),
        "macro_f1": round(f1_score([x["gold"] for x in res], [x["pred"] for x in res], average="macro", zero_division=0), 3),
        "accuracy": round(sum(x["gold"] == x["pred"] for x in res) / len(res), 3),
        f"recall@{TOP_K}": rate("hit", ev),
        f"full_set_recall@{TOP_K}": rate("full_hit", ev),
        "mrr": rate("rr", ev),
        "grounded_accuracy": rate("grounded", res),
    }


everything = [x for res in by_lang.values() for x in res]
report = {"overall": summary(everything), **{lang: summary(res) for lang, res in by_lang.items()}}
en_f1 = report.get("en", {}).get("macro_f1")
for lang, s in report.items():
    if lang not in ("overall", "en") and en_f1:
        s["f1_change_vs_en"] = round(s["macro_f1"] - en_f1, 3)
report["overall"]["sec_per_claim"] = round((time.perf_counter() - t0) / len(rows), 2)

print(json.dumps(report, indent=2))
print(classification_report([x["gold"] for x in everything], [x["pred"] for x in everything], zero_division=0))
Path("results").mkdir(exist_ok=True)
Path(f"results/pipeline_{path.stem}.json").write_text(json.dumps(report, indent=2))
