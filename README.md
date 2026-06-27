# Redrob Hackathon — Candidate Ranker

Ranks the 100K-candidate pool for the "Senior AI Engineer — Founding Team" JD
and produces a top-100 CSV per `submission_spec.md`.

## Why this approach

The JD is explicit that the trap is "keyword count == fit." A Tier-5
candidate may never write "RAG" or "Pinecone" in their skills list but still
be the right hire if their career history shows they shipped a ranking/search
system at a product company. A candidate with every AI buzzword in their
skills section but a "Marketing Manager" title is not a fit.

So this ranker does **not** do bag-of-skills keyword matching. It scores five
independent dimensions per candidate, built from regex/keyword scans over
**career history descriptions**, not just the skills array:

| Dimension | What it captures | Weight |
|---|---|---|
| Technical depth | production embeddings/retrieval, vector-DB/hybrid search, ranking systems, eval frameworks (NDCG/MRR/MAP/AB), Python, evidence the work shipped to production | 40% |
| Title relevance | is the *current title* actually AI/ML/search-flavored, vs. adjacent, vs. unrelated | 30% |
| Logistics fit | Pune/Noida/Tier-1 location or relocation, notice period, years-of-experience band (5–9, soft falloff outside) | 20% |
| Disqualifier multiplier | the JD's explicit "don't wants": pure-research-only, recent-LangChain-only AI experience, architecture/lead titles with no recent hands-on coding, job-hopping across rising titles, pure-consulting-only career, CV/speech/robotics without NLP exposure | multiplicative penalty, 0.25×–1.0× |
| Behavioral modifier | recency of activity, recruiter response rate, open-to-work flag, interview completion rate, offer acceptance rate — modeled as a *modifier* on top of fit, not a core driver, per the JD's own framing | scales final score 0.7×–1.0× |

`final_score = (0.40·technical + 0.30·title + 0.20·logistics) × disqualifier_multiplier × (0.7 + 0.3·behavioral)`

### Honeypot handling

Before scoring, every candidate is checked for internally-impossible profiles:
"expert" proficiency in a skill used for ~0 months, overlapping concurrent
roles, `end_date` before `start_date`, and a career-history total tenure that
doesn't plausibly match the stated `years_of_experience`. Flagged profiles
are forced to a near-zero score (not removed — they still appear in the pool,
they just sink) rather than special-cased into the scoring logic itself, so
the system is "naturally avoiding them" as the brief asks for, not gaming the
specific honeypot construction.

### Why no embeddings / no LLM calls

The compute budget (5 min, 16GB RAM, CPU-only, no network) rules out an
LLM-per-candidate approach outright, and the brief is explicit that this is
the point: a system has to actually think about latency/quality tradeoffs.
Regex/keyword scans over career-history text are O(n), fully local, and
score 100K candidates in ~100 seconds on a laptop CPU — and every score is
traceable to a one-line, non-hallucinated reason, which is what Stage 4
manual review is checking for.

## Repo layout

```
rank.py              # entry point
src/scoring.py        # all scoring logic (the actual "model")
requirements.txt
submission_metadata.yaml
candidates.jsonl       # (not committed — see Setup)
```

## Setup

```bash
pip install -r requirements.txt   # stdlib only today, kept for future deps
```

No model weights, no API keys, no pre-computation step. The script is the
entire pipeline.

## Reproduce the submission CSV

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Tested at 100,000 candidates in ~100 seconds on a CPU-only laptop, well
inside the 5-minute budget. `--limit N` is available for quick local
debugging on a subset.

## Validate

```bash
python validate_submission.py submission.csv
```

## Known limitations / honest caveats

- Regex-based skill/keyword detection misses paraphrases the lexicons don't
  anticipate (e.g. an unusual term for "vector database"). This is a
  precision/recall tradeoff made deliberately in favor of explainability and
  speed over an embedding-based semantic match, given the no-GPU/no-network
  constraint.
- The disqualifier penalties (e.g. "pure consulting only", "job-hopping")
  are heuristics built directly from the JD's stated dislikes, not learned
  from labeled data — there's no ground truth available to fit weights
  against, so weights were chosen by reasoning through the JD text and spot
  checking the top-100 output (see `methodology.md` for the spot-checks run).
- This is a single ranking pass; no learning-to-rank model is trained
  because there is no labeled training signal provided in the dataset.
