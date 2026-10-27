"""Traditional baseline: TF-IDF + LogisticRegression on claim text only, trained on FEVER train.

Claim-only by design: it sees no evidence, so it measures how much the label leaks from claim wording
(FEVER's known "claim-only bias"). Any score above chance here is not evidence-grounded verification.
"""
import json
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline

D = Path("data/fever")
LABELS = {"SUPPORTS": "SUPPORTED", "REFUTES": "REFUTED", "NOT ENOUGH INFO": "NOT_ENOUGH_EVIDENCE"}


def load(p, raw_labels=True):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return [r["claim"] for r in rows], [LABELS[r["label"]] if raw_labels else r["label"] for r in rows]


Xtr, ytr = load(D / "train.jsonl")
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
).fit(Xtr, ytr)

report = {"train_n": len(Xtr)}
sample = Path(sys.argv[1] if len(sys.argv) > 1 else D / "dev_sample.jsonl")  # same file eval.py scores
for name, (X, y) in {sample.stem: load(sample, raw_labels=False),
                     "paper_dev_full": load(D / "paper_dev.jsonl")}.items():
    pred = model.predict(X)
    report[name] = {"n": len(y), "macro_f1": round(f1_score(y, pred, average="macro"), 3),
                    "accuracy": round(accuracy_score(y, pred), 3)}
    print(f"== {name} ==\n" + classification_report(y, pred, zero_division=0))

print(json.dumps(report, indent=2))
Path("results").mkdir(exist_ok=True)
Path("results/baseline_tfidf_lr.json").write_text(json.dumps(report, indent=2))
