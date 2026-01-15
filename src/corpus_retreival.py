import re
import json
import math
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from typing import Optional, Set, List, Tuple, Union, Dict


# ---------------------------
# 1) Question loading
# ---------------------------
def load_questions(csv_path: str, question_col: str = "en_question") -> List[str]:
    df = pd.read_csv(csv_path)
    if question_col not in df.columns:
        raise ValueError(
            f"Column '{question_col}' not found in {csv_path}. Available: {list(df.columns)}"
        )
    return df[question_col].dropna().astype(str).tolist()


def extract_question_from_mcq_prompt(prompt: str) -> str:
    """
    Keep only the actual question, remove the repeated template/options.
    """
    prompt = re.split(r"without any explanation", prompt, flags=re.IGNORECASE)[0]
    prompt = re.split(r"\n\s*A\.", prompt)[0]
    prompt = re.split(r"\n\s*Answer\s*:", prompt, flags=re.IGNORECASE)[0]
    return prompt.strip()


# ---------------------------
# 1b) Country codes + names
# ---------------------------
# Minimal mapping for common codes -> canonical country names for Wikipedia.
# Add more as needed based on your dataset.
CODE_TO_NAME: Dict[str, str] = {
    "US": "United States",
    "UK": "United Kingdom",
    "GB": "United Kingdom",
    "CN": "China",
    "IR": "Iran",
}


def collect_country_codes(*csv_specs: Tuple[str, str]) -> Set[str]:
    """
    csv_specs: tuples of (csv_path, country_column_name)
    returns set of codes like {"US","GB","CN","IR",...}
    """
    codes: Set[str] = set()
    for path, col in csv_specs:
        df = pd.read_csv(path)
        if col in df.columns:
            vals = df[col].dropna().astype(str).str.strip().str.upper().tolist()
            # keep only short alphabetic codes to avoid things like "IRAN", "CHINA"
            codes.update(v for v in vals if 2 <= len(v) <= 3 and v.isalpha())
    return codes


def collect_country_names_from_columns(*csv_specs: Tuple[str, str]) -> Set[str]:
    """
    Collect country names from columns that contain names (e.g. MCQ 'country' column has 'Iran', 'US', etc.).
    We keep values that look like names (len>3) and also accept ISO codes if present.
    """
    names: Set[str] = set()
    for path, col in csv_specs:
        df = pd.read_csv(path)
        if col in df.columns:
            vals = df[col].dropna().astype(str).str.strip()
            for v in vals:
                if not v:
                    continue
                v_up = v.upper()
                # if it's a known code, map to name
                if 2 <= len(v_up) <= 3 and v_up.isalpha() and v_up in CODE_TO_NAME:
                    names.add(CODE_TO_NAME[v_up])
                # if it looks like a name, keep it
                elif len(v) > 3:
                    names.add(v)
    return names


def expand_questions_with_country_names(
    questions: List[str],
    country_names: Set[str],
) -> List[str]:
    """
    Add country NAMES as extra "pseudo-questions" so TF-IDF picks them up
    (and so your Wikipedia filter matches pages titled by country name).
    """
    out = list(questions)
    out.extend(sorted(country_names))
    return out


# ---------------------------
# 2) Keyword extraction (TF-IDF, still lightweight)
# ---------------------------
STOPWORDS = set(
    """
a an the and or of in on at to for with without from by as is are was were be been being
this that these those it its they them their you your we our i me my
which what when where who whom why how
""".split()
)


def normalize_text(s: str) -> str:
    s = s.lower()
    # keep unicode letters/numbers/underscore/hyphen, replace other punctuation with spaces
    s = re.sub(r"[^\w\s\-]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_keywords(
    questions: List[str],
    top_k: int = 300,
    min_df: int = 2,
    allow_2char: Optional[Set[str]] = None,
    return_scored: bool = False,
) -> Union[Set[str], Tuple[Set[str], List[Tuple[str, float, int, int]]]]:
    """
    TF-IDF keyword selection over questions.

    - Each question is a "document"
    - DF: in how many questions a token appears
    - TF: total token count across questions
    - IDF = log((N + 1)/(df + 1)) + 1
    - score = TF * IDF
    - keep top_k by score, filter tokens with df < min_df

    Special handling:
    - 2-letter tokens (US, GB, CN, IR...) are normally filtered out,
      but allowed if they appear in allow_2char.

    Returns:
      - keywords (Set[str]) if return_scored=False
      - (keywords, scored) if return_scored=True, where scored is:
            List[(term, score, tf_count, df_count)]
    """
    allow_2char = {c.upper() for c in (allow_2char or set())}

    N = len(questions)
    if N == 0:
        return (set(), []) if return_scored else set()

    tf: Dict[str, int] = {}
    df: Dict[str, int] = {}

    for q in questions:
        qn = normalize_text(q)
        tokens: List[str] = []

        for t in qn.split():
            t_up = t.upper()

            # allow 2-char tokens only if whitelisted
            if len(t) < 3 and t_up not in allow_2char:
                continue

            if t in STOPWORDS:
                continue

            tokens.append(t)
            tf[t] = tf.get(t, 0) + 1

        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    scored: List[Tuple[str, float, int, int]] = []
    for term, tf_count in tf.items():
        term_df = df.get(term, 0)
        if term_df < min_df:
            continue
        idf = math.log((N + 1) / (term_df + 1)) + 1.0
        score = tf_count * idf
        scored.append((term, score, tf_count, term_df))

    # sort by score desc
    scored.sort(key=lambda x: x[1], reverse=True)

    keep = {term for term, _, _, _ in scored[:top_k]}
    if return_scored:
        return keep, scored
    return keep


# ---------------------------
# 3) Chunking
# ---------------------------
def chunk_text(text: str, chunk_words: int = 220, overlap_words: int = 60) -> List[str]:
    words = text.split()
    if len(words) <= chunk_words:
        return [" ".join(words)]

    chunks: List[str] = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        end = start + chunk_words
        chunk = words[start:end]
        if len(chunk) < 60:  # drop tiny tail chunks
            break
        chunks.append(" ".join(chunk))
    return chunks


# ---------------------------
# 4) Wikipedia streaming + filtering + output
# ---------------------------
def build_subset(
    out_jsonl: str,
    keywords: Set[str],
    wiki_lang: str = "en",
    max_pages: int = 30000,     # hard cap for storage control
    max_chunks: int = 150000,   # hard cap for storage control
    chunk_words: int = 220,
    overlap_words: int = 60,
    min_text_words: int = 120,
    snapshot: str = "20231101",
):
    """
    Streams Wikipedia and keeps a page if it matches the keyword set.
    Writes out JSONL chunks with title + text.

    NOTE: This is a broad filter: it matches if *any* keyword appears in (title + first 800 chars).
    If your keywords include common words, you'll get a lot of unrelated pages.
    """
    config = f"{snapshot}.{wiki_lang}"
    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)

    kept_pages = 0
    written_chunks = 0
    chunk_id = 0

    if not keywords:
        raise ValueError("Keyword set is empty. Check your question file/column or keyword extraction params.")

    # IMPORTANT: escape keywords for regex safety
    kw_pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted(keywords))) + r")\b", flags=re.IGNORECASE)

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="Streaming Wikipedia"):
            if kept_pages >= max_pages or written_chunks >= max_chunks:
                break

            title = row.get("title") or ""
            text = row.get("text") or ""
            if not title or not text:
                continue

            if len(text.split()) < min_text_words:
                continue

            # probe on title + a snippet for speed
            probe = normalize_text(title + " " + text[:800])

            if not kw_pattern.search(probe):
                continue

            kept_pages += 1

            chunks = chunk_text(text, chunk_words=chunk_words, overlap_words=overlap_words)
            for c in chunks:
                if written_chunks >= max_chunks:
                    break

                chunk_id += 1
                obj = {
                    "id": chunk_id,
                    "title": title,
                    "text": c,
                    "source": f"wikipedia:{wiki_lang}:{snapshot}",
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written_chunks += 1

    print(f"\nDone. Kept pages: {kept_pages:,} | Written chunks: {written_chunks:,}")
    print(f"Output: {out_jsonl}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Load training questions
    print("started")
    saq_qs = load_questions("data/train_dataset_saq.csv", question_col="en_question")

    mcq_prompts = load_questions("data/train_dataset_mcq.csv", question_col="prompt")
    mcq_qs = [extract_question_from_mcq_prompt(p) for p in mcq_prompts]

    questions = saq_qs + mcq_qs

    # ---- Fix 1: allow 2-letter country codes (US/GB/CN/IR/...) from SAQ country col
    allow_2char = collect_country_codes(
        ("data/train_dataset_saq.csv", "country"),
    )

    # ---- Fix 2: add country NAMES as extra "pseudo-questions"
    country_names: Set[str] = set()
    for code in allow_2char:
        if code in CODE_TO_NAME:
            country_names.add(CODE_TO_NAME[code])

    # MCQ has a 'country' column (names like Iran, US, etc.)
    country_names |= collect_country_names_from_columns(
        ("data/train_dataset_mcq.csv", "country"),
    )

    questions = expand_questions_with_country_names(questions, country_names)

    # TF-IDF keywords
    keywords, scored = extract_keywords(
        questions,
        top_k=300,
        min_df=2,
        allow_2char=allow_2char,
        return_scored=True,
    )

    print("TOP 50 by score:")
    for term, score, tfc, dfi in scored[:50]:
        print(term, round(score, 4), tfc, dfi)

    print(f"\nKeyword vocab size: {len(keywords)}")
    print(f"Example country codes kept: {sorted(list(allow_2char))[:20]}")
    print(f"Example country names added: {sorted(list(country_names))[:20]}")

    # Build subset
    build_subset(
        out_jsonl="wiki_chunks.jsonl",
        keywords=keywords,
        wiki_lang="en",
        snapshot="20231101",
        max_pages=30000,
        max_chunks=150000,
        chunk_words=220,
        overlap_words=60,
    )
