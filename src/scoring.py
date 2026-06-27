"""
Scoring engine for the Redrob "Senior AI Engineer — Founding Team" JD.

Design philosophy (see README for the long version):
  The JD explicitly says the trap is "keyword count == fit". So instead of a
  bag-of-skills score, we score five independent, inspectable dimensions and
  combine them. Every dimension is built from regex/keyword lookups over
  *career_history descriptions* (not just the skills list), because the JD
  explicitly rewards candidates who "built a recommendation system at a
  product company" even if they never wrote "RAG" in their skills section.

  Disqualifiers from the JD ("things we explicitly do NOT want") are modeled
  as multiplicative penalties, not hard zeroes -- except honeypots, which are
  forced to ~0 because the ground truth forces them to relevance tier 0.

Everything here is pure-Python / regex -- no embeddings, no model downloads,
no GPU, no network. This is intentional: it is what lets the whole 100K-row
pool be ranked in well under the 5-minute / 16GB / CPU-only budget, and it
keeps every score traceable to a one-line reason (Stage 4 reasoning review).
"""

import re
import datetime
from statistics import mean

TODAY = datetime.date(2026, 6, 27)

# ---------------------------------------------------------------------------
# Keyword lexicons (compiled once, reused 100K times)
# ---------------------------------------------------------------------------

def _rx(words):
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.I)

EMBEDDING_RETRIEVAL = _rx([
    "embedding", "embeddings", "sentence-transformers", "sentence transformers",
    "bge", "e5", "openai embeddings", "dense retrieval", "semantic search",
    "vector search", "ann search", "approximate nearest neighbor",
])

VECTOR_DB = _rx([
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "faiss", "vector database", "vector db", "hybrid search", "bm25",
])

RANKING_SYSTEMS = _rx([
    "ranking system", "re-ranking", "reranking", "recommendation system",
    "recommender system", "search ranking", "learning to rank",
    "learning-to-rank", "candidate ranking", "relevance ranking",
    "matching system", "retrieval system",
])

LLM_WORK = _rx([
    "llm", "large language model", "fine-tun", "lora", "qlora", "peft",
    "rag", "retrieval augmented", "retrieval-augmented", "prompt engineering",
    "gpt", "transformer model",
])

EVAL_FRAMEWORKS = _rx([
    "ndcg", "mrr", "map@", "mean average precision", "precision@", "recall@",
    "a/b test", "ab test", "offline.{0,15}online", "online.{0,15}offline",
    "evaluation framework", "offline benchmark", "click-through",
])

PRODUCTION_SIGNAL = _rx([
    "production", "deployed", "real users", "at scale", "live traffic",
    "shipped", "in prod", "serving traffic", "millions of", "real-time",
])

PRE_LLM_ML = _rx([
    "tf-idf", "word2vec", "glove", "logistic regression", "xgboost",
    "gradient boost", "random forest", "click model", "collaborative filtering",
    "feature store", "search relevance", "information retrieval",
])

NLP_IR = _rx([
    "nlp", "natural language", "information retrieval", "text classification",
    "named entity", "search relevance", "query understanding", "embeddings",
    "semantic search", "language model", "text mining",
])

CV_SPEECH_ROBOTICS_ONLY = _rx([
    "computer vision", "image classification", "object detection",
    "speech recognition", "speech-to-text", "robotics", "slam",
    "autonomous navigation", "image segmentation",
])

PYTHON_SIGNAL = _rx(["python"])

FRAMEWORK_TUTORIAL_SMELL = _rx([
    "langchain tutorial", "built a demo", "how i used", "weekend project",
    "followed a tutorial", "toy project",
])

ARCHITECTURE_ONLY_TITLES = _rx([
    "architect", "tech lead", "engineering manager", "principal engineer",
    "director", "vp of engineering", "head of engineering",
])

IC_TITLES = _rx([
    "engineer", "developer", "scientist", "researcher", "programmer",
])

RESEARCH_ONLY_INDUSTRY = _rx([
    "research", "academia", "university", "academic",
])

PURE_SERVICES_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree", "ltimindtree",
    "genpact", "ibm services",
}

TIER1_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr", "gurugram",
    "gurgaon", "bangalore", "bengaluru", "chennai", "new delhi",
}
PRIMARY_CITIES = {"pune", "noida"}


def _years_between(start, end):
    try:
        s = datetime.date.fromisoformat(start)
    except Exception:
        return 0
    e = TODAY if not end else _safe_date(end)
    return max(0.0, (e - s).days / 365.25)


def _safe_date(s):
    try:
        return datetime.date.fromisoformat(s)
    except Exception:
        return TODAY


def _text_blob(candidate):
    """All free-text fields, concatenated once for keyword scanning."""
    parts = [candidate["profile"].get("headline", ""),
              candidate["profile"].get("summary", "")]
    for job in candidate.get("career_history", []):
        parts.append(job.get("description", ""))
        parts.append(job.get("title", ""))
    return " \n ".join(parts)


def _skill_names(candidate):
    return {s["name"].strip().lower(): s for s in candidate.get("skills", [])}


# ---------------------------------------------------------------------------
# Honeypot / impossible-profile detection
# ---------------------------------------------------------------------------

def detect_honeypot(candidate):
    reasons = []

    for s in candidate.get("skills", []):
        if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) <= 1:
            reasons.append(f"'expert' in {s['name']} with ~0 months used")

    history = candidate.get("career_history", [])
    current_flags = sum(1 for j in history if j.get("is_current"))
    if current_flags > 1:
        reasons.append("multiple concurrent 'current' roles")

    for j in history:
        sd, ed = j.get("start_date"), j.get("end_date")
        if sd and ed:
            try:
                if datetime.date.fromisoformat(ed) < datetime.date.fromisoformat(sd):
                    reasons.append(f"end_date before start_date at {j.get('company')}")
            except Exception:
                pass

    total_months = sum(j.get("duration_months", 0) or 0 for j in history)
    yoe = candidate["profile"].get("years_of_experience", 0) or 0
    if yoe > 0:
        ratio = (total_months / 12.0) / yoe
        if ratio < 0.4 or ratio > 2.2:
            reasons.append(
                f"career_history totals {total_months}mo vs stated {yoe}yrs experience"
            )

    # earliest job started before candidate could plausibly have any experience
    if history:
        earliest = min(_safe_date(j["start_date"]) for j in history if j.get("start_date"))
        span_years = (TODAY - earliest).days / 365.25
        if yoe > 0 and span_years < yoe * 0.5:
            reasons.append("career span far shorter than claimed years_of_experience")

    return reasons  # non-empty => honeypot


# ---------------------------------------------------------------------------
# Dimension scores (each 0-1ish before weighting)
# ---------------------------------------------------------------------------

def score_title_relevance(candidate):
    title = candidate["profile"].get("current_title", "")
    if ARCHITECTURE_ONLY_TITLES.search(title) and not IC_TITLES.search(title):
        return 0.55, "title suggests architecture/management, not hands-on IC work"
    relevant = _rx([
        "ai", "ml", "machine learning", "data scien", "research engineer",
        "applied scientist", "search", "ranking", "recommend",
    ])
    if relevant.search(title):
        return 1.0, f"current title ('{title}') is directly in the AI/ML/search lane"
    adjacent = _rx(["backend", "data engineer", "software engineer", "full stack"])
    if adjacent.search(title):
        return 0.6, f"current title ('{title}') is adjacent engineering, not AI-titled"
    return 0.15, f"current title ('{title}') is unrelated to AI/ML engineering"


def score_technical_depth(candidate):
    blob = _text_blob(candidate)
    skills = _skill_names(candidate)

    has_embed = bool(EMBEDDING_RETRIEVAL.search(blob)) or any(
        k in skills for k in ["embeddings", "sentence-transformers", "bge", "e5"])
    has_vecdb = bool(VECTOR_DB.search(blob)) or any(
        k in skills for k in ["pinecone", "weaviate", "qdrant", "milvus",
                                "elasticsearch", "opensearch", "faiss"])
    has_ranking = bool(RANKING_SYSTEMS.search(blob))
    has_eval = bool(EVAL_FRAMEWORKS.search(blob))
    has_prod = bool(PRODUCTION_SIGNAL.search(blob))
    has_python = ("python" in skills) or bool(PYTHON_SIGNAL.search(blob))

    must_haves = [has_embed, has_vecdb, has_python]
    must_score = sum(must_haves) / len(must_haves)

    nice = [has_ranking, has_eval, has_prod]
    nice_score = sum(nice) / len(nice)

    score = 0.7 * must_score + 0.3 * nice_score

    bits = []
    if has_embed:
        bits.append("production embeddings/retrieval")
    if has_vecdb:
        bits.append("vector-DB/hybrid-search infra")
    if has_eval:
        bits.append("ranking-evaluation (NDCG/MRR/MAP/AB)")
    if has_prod:
        bits.append("shipped to production")
    reason = ("; ".join(bits) if bits else "no embeddings/vector-DB/eval signal found")
    return score, reason


def score_disqualifiers(candidate):
    """Returns a multiplier in (0, 1] and a list of reasons applied."""
    blob = _text_blob(candidate)
    skills = _skill_names(candidate)
    history = candidate.get("career_history", [])
    industry = candidate["profile"].get("current_industry", "")
    mult = 1.0
    reasons = []

    # Pure research-only, no production deployment
    if RESEARCH_ONLY_INDUSTRY.search(industry) and not PRODUCTION_SIGNAL.search(blob):
        mult *= 0.25
        reasons.append("research-only background with no production deployment evidence")

    # Recent LangChain/OpenAI-only "AI experience"
    llm_only_recent = False
    for s in skills.values():
        if s["name"].lower() in {"langchain", "openai api"} and (s.get("duration_months") or 99) < 12:
            llm_only_recent = True
    if llm_only_recent and not PRE_LLM_ML.search(blob):
        mult *= 0.5
        reasons.append("AI experience looks LangChain/OpenAI-only and <12mo, no pre-LLM ML evidence")

    # Architecture/manager role with no recent coding signal
    title = candidate["profile"].get("current_title", "")
    if ARCHITECTURE_ONLY_TITLES.search(title):
        current_job = next((j for j in history if j.get("is_current")), None)
        months_in_role = current_job.get("duration_months", 0) if current_job else 0
        if months_in_role >= 18 and not PRODUCTION_SIGNAL.search(
            current_job.get("description", "") if current_job else ""
        ):
            mult *= 0.55
            reasons.append("18+ months in an architecture/lead title with no recent hands-on coding signal")

    # Title-chasing: 3+ jobs, rising seniority words, avg tenure < 18mo
    if len(history) >= 3:
        avg_tenure = mean(j.get("duration_months", 0) or 0 for j in history)
        seniority_words = _rx(["senior", "staff", "principal", "lead"])
        rising = sum(1 for j in history if seniority_words.search(j.get("title", "")))
        if avg_tenure < 18 and rising >= 2:
            mult *= 0.6
            reasons.append("job-hopping pattern across rising titles, avg tenure <18 months")

    # Framework-tutorial smell
    if FRAMEWORK_TUTORIAL_SMELL.search(blob):
        mult *= 0.85
        reasons.append("profile language reads as tutorial/demo-driven rather than systems-driven")

    # Pure consulting-only career
    companies = {j.get("company", "").strip().lower() for j in history}
    companies.add(candidate["profile"].get("current_company", "").strip().lower())
    if companies and companies.issubset(PURE_SERVICES_FIRMS):
        mult *= 0.45
        reasons.append("entire career inside services/consulting firms (no product-company experience)")

    # CV/speech/robotics-only without NLP/IR
    if CV_SPEECH_ROBOTICS_ONLY.search(blob) and not NLP_IR.search(blob):
        mult *= 0.5
        reasons.append("CV/speech/robotics background without NLP/IR exposure")

    return mult, reasons


def score_logistics(candidate):
    profile = candidate["profile"]
    loc = (profile.get("location") or "").strip().lower()
    relocate = candidate["redrob_signals"].get("willing_to_relocate", False)
    notice = candidate["redrob_signals"].get("notice_period_days", 60)

    loc_score = 0.4
    loc_reason = f"location ({profile.get('location')}) not Tier-1 and not open to relocating"
    if any(c in loc for c in PRIMARY_CITIES):
        loc_score, loc_reason = 1.0, f"based in {profile.get('location')} (primary office city)"
    elif any(c in loc for c in TIER1_CITIES):
        loc_score, loc_reason = 0.85, f"based in Tier-1 city ({profile.get('location')})"
    elif relocate:
        loc_score, loc_reason = 0.7, "not in a target city but open to relocation"

    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.7
    else:
        notice_score = 0.45

    yoe = profile.get("years_of_experience", 0) or 0
    if 5 <= yoe <= 9:
        yoe_score = 1.0
    else:
        dist = min(abs(yoe - 5), abs(yoe - 9))
        yoe_score = max(0.3, 1.0 - dist * 0.12)

    score = 0.4 * loc_score + 0.25 * notice_score + 0.35 * yoe_score
    reason = f"{loc_reason}; notice {notice}d; {yoe}yrs experience"
    return score, reason


def score_behavioral(candidate):
    sig = candidate["redrob_signals"]
    last_active = _safe_date(sig.get("last_active_date", str(TODAY)))
    days_inactive = (TODAY - last_active).days
    recency = max(0.0, 1.0 - days_inactive / 180.0)

    response = sig.get("recruiter_response_rate", 0.0) or 0.0
    open_flag = 1.0 if sig.get("open_to_work_flag") else 0.5
    interview_rate = sig.get("interview_completion_rate", 0.5) or 0.5
    oar = sig.get("offer_acceptance_rate", 0.0)
    offer_term = 0.5 if oar is None or oar < 0 else oar

    score = (
        0.30 * recency + 0.30 * response + 0.15 * open_flag
        + 0.15 * interview_rate + 0.10 * offer_term
    )
    if days_inactive > 180:
        reason = f"inactive {days_inactive}d, response rate {response:.0%} -- likely unreachable"
    else:
        reason = f"active {days_inactive}d ago, recruiter response rate {response:.0%}"
    return score, reason


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def score_candidate(candidate):
    honeypot_reasons = detect_honeypot(candidate)
    if honeypot_reasons:
        return {
            "score": 0.001,
            "reason": "Flagged as a likely honeypot/impossible profile (" + honeypot_reasons[0] + "); excluded from consideration.",
            "is_honeypot": True,
        }

    title_s, title_r = score_title_relevance(candidate)
    tech_s, tech_r = score_technical_depth(candidate)
    dq_mult, dq_reasons = score_disqualifiers(candidate)
    logi_s, logi_r = score_logistics(candidate)
    behav_s, behav_r = score_behavioral(candidate)

    base = 0.40 * tech_s + 0.30 * title_s + 0.20 * logi_s
    base *= dq_mult
    final = base * (0.7 + 0.3 * behav_s)  # behavioral as modifier, not core driver
    final = max(0.0, min(1.0, final))

    name = candidate["profile"].get("anonymized_name", "Candidate")
    title = candidate["profile"].get("current_title", "")
    yoe = candidate["profile"].get("years_of_experience", 0)
    reason_bits = [f"{title}, {yoe:.1f} yrs"]
    reason_bits.append(tech_r)
    if dq_reasons:
        reason_bits.append("concern: " + dq_reasons[0])
    reason_bits.append(behav_r)
    reasoning = "; ".join(reason_bits)
    if len(reasoning) > 280:
        reasoning = reasoning[:277] + "..."

    return {
        "score": round(final, 4),
        "reason": reasoning,
        "is_honeypot": False,
    }
