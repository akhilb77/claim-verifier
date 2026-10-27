"""Build a FEVER sentence corpus + dev sample from data/fever/{paper_dev.jsonl,wiki-pages.zip}.

Corpus = every page cited as evidence by the sampled dev claims, plus ~1/1000 of all other pages as distractors.
ponytail: not the full 5.4M-page wiki — encoding that on CPU takes days. Retrieval here is easier than open
FEVER; raise DISTRACTOR_PER_MILLE (or use FAISS + GPU over the whole dump) for a harder, honest setting.
"""
import json
import random
import sys
import zipfile
import zlib
from pathlib import Path

D = Path("data/fever")
N_DEV = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
DISTRACTOR_PER_MILLE = 1
LABELS = {"SUPPORTS": "SUPPORTED", "REFUTES": "REFUTED", "NOT ENOUGH INFO": "NOT_ENOUGH_EVIDENCE"}
UNESCAPE = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}", "-COLON-": ":"}


def unescape(s: str) -> str:
    for k, v in UNESCAPE.items():
        s = s.replace(k, v)
    return s


dev = [json.loads(l) for l in open(D / "paper_dev.jsonl", encoding="utf-8")]
random.Random(0).shuffle(dev)
dev = dev[:N_DEV]

sample, wanted = [], set()
for r in dev:
    sets = None
    if r["label"] != "NOT ENOUGH INFO":
        sets = sorted({tuple(sorted(f"{e[2]}:{e[3]}" for e in s)) for s in r["evidence"]})
        wanted |= {e[2] for s in r["evidence"] for e in s}
    sample.append({"claim": r["claim"], "language": "en", "label": LABELS[r["label"]], "gold": [list(s) for s in sets] if sets else None})

n_pages = n_sents = 0
with zipfile.ZipFile(D / "wiki-pages.zip") as z, open(D / "corpus.jsonl", "w", encoding="utf-8") as out:
    for name in sorted(n for n in z.namelist() if n.startswith("wiki-pages/") and n.endswith(".jsonl")):  # skip __MACOSX junk
        for line in z.open(name):
            page = json.loads(line)
            pid = page["id"]
            if not page["lines"] or (pid not in wanted and zlib.crc32(pid.encode()) % 1000 >= DISTRACTOR_PER_MILLE):
                continue
            title = unescape(pid).replace("_", " ")
            n_pages += 1
            for row in page["lines"].split("\n"):
                parts = row.split("\t")
                if len(parts) < 2 or not parts[1].strip():
                    continue
                n_sents += 1
                out.write(json.dumps({
                    "id": f"{pid}:{parts[0]}",
                    # prepend the title: FEVER sentences often say "He ..." without naming the subject
                    "text": f"{title}: {unescape(parts[1])}",
                    "source_title": f"{title} (Wikipedia, sentence {parts[0]})",
                    "source_url": "https://en.wikipedia.org/wiki/" + unescape(pid),
                    "language": "en",
                    "provenance": "FEVER 2017 Wikipedia dump; the live article may have changed",
                }, ensure_ascii=False) + "\n")
        print(f"{name}: {n_pages} pages, {n_sents} sentences", flush=True)

with open(D / "dev_sample.jsonl", "w", encoding="utf-8") as f:
    f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in sample)
print(f"done: {len(sample)} dev claims, {n_pages} pages ({len(wanted)} gold), {n_sents} sentences")
