#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon ranker.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Reads the full candidate pool (plain or gzipped JSONL), scores every
candidate with src/scoring.py (pure Python, no network/GPU), and writes the
top-100 ranked CSV in the exact format required by submission_spec.md.

Designed to finish well inside the 5-minute / 16GB / CPU-only budget for a
100K-candidate pool: O(n) single pass, regex precompiled once, no model
loading.
"""

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from scoring import score_candidate  # noqa: E402

TOP_N = 100


def open_candidates(path):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, "r", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="debug: only score first N rows")
    args = ap.parse_args()

    t0 = time.time()
    results = []
    honeypot_count = 0
    n = 0

    with open_candidates(args.candidates) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            cand = json.loads(line)
            try:
                r = score_candidate(cand)
            except Exception as e:  # never let one bad row kill the run
                r = {"score": 0.0, "reason": f"scoring error: {e}", "is_honeypot": False}
            if r["is_honeypot"]:
                honeypot_count += 1
            results.append((cand["candidate_id"], r["score"], r["reason"]))
            if args.limit and n >= args.limit:
                break

    # Sort: score desc, candidate_id asc as deterministic tiebreak
    results.sort(key=lambda x: (-x[1], x[0]))
    top = results[:TOP_N]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, (cid, score, reason) in enumerate(top, start=1):
            w.writerow([cid, i, f"{score:.4f}", reason])

    elapsed = time.time() - t0
    top_honeypots = sum(1 for cid, s, r in top if "honeypot" in r.lower())
    print(f"Scored {n} candidates in {elapsed:.1f}s.")
    print(f"Honeypots detected overall: {honeypot_count}")
    print(f"Honeypots in top {TOP_N}: {top_honeypots} ({top_honeypots/TOP_N:.1%})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
