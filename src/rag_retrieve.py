# rag_retrieve.py
# Dense retrieval helper used at inference time (import this from predict.py).
#
# Works with your saved files:
#   output/rag_index/faiss.index
#   output/rag_index/meta.jsonl
#
# Assumes:
# - The FAISS index rows correspond 1:1 (same order) with lines in meta.jsonl
# - Index was built using L2-normalized embeddings (common for cosine via Inner Product)
# - You want E5-style prefixes: "query: ..." and "passage: ..."
#
# If your index was built WITHOUT normalization, set normalize_query=False when calling.

import json
from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_meta(meta_path: str) -> List[Dict]:
    """
    Loads metadata as JSONL (one JSON object per line).

    Expected fields per line (example):
      {"id": 123, "title": "...", "text": "...", "source": "..."}
    """
    meta: List[Dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta.append(json.loads(line))
    return meta


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # x: (n, d)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


class DenseRetriever:
    """
    FAISS + SentenceTransformer retriever.

    Changeable parameters:
      - index_path: path to faiss.index
      - meta_path: path to meta.jsonl
      - embed_model: e5-base-v2 / e5-large-v2 / etc.
      - device: "cuda" or "cpu"
      - query_prefix / passage_prefix: E5 uses "query:" and "passage:"
      - normalize_query: True if index uses cosine/IP with normalized vectors
      - max_query_chars: truncate very long queries (safety)
    """

    def __init__(
        self,
        index_path: str = "output/rag_index/faiss.index",
        meta_path: str = "output/rag_index/meta.jsonl",
        embed_model: str = "intfloat/e5-large-v2",
        device: str = "cuda",
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",  # kept here for consistency / debugging
        normalize_query: bool = True,
        max_query_chars: int = 2000,
    ):
        self.index_path = index_path
        self.meta_path = meta_path

        self.index = faiss.read_index(index_path)
        self.meta = load_meta(meta_path)

        # Basic sanity check: index size should match meta length
        # If not, retrieval still "works" but mappings will be wrong.
        ntotal = getattr(self.index, "ntotal", None)
        if ntotal is not None and ntotal != len(self.meta):
            raise ValueError(
                f"FAISS index ntotal ({ntotal}) != meta lines ({len(self.meta)}). "
                f"These must match 1:1 in the same order."
            )

        self.model = SentenceTransformer(embed_model, device=device)

        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.normalize_query = normalize_query
        self.max_query_chars = max_query_chars

    def embed_query(self, query: str) -> np.ndarray:
        q = (query or "").strip()
        if len(q) > self.max_query_chars:
            q = q[: self.max_query_chars]

        q = f"{self.query_prefix}{q}"

        q_emb = self.model.encode(
            [q],
            convert_to_numpy=True,
            normalize_embeddings=False,  # we control normalization ourselves
            show_progress_bar=False,
        ).astype("float32")

        if self.normalize_query:
            q_emb = l2_normalize(q_emb)

        return q_emb

    def search(self, query: str, k: int = 5) -> List[Tuple[float, Dict]]:
        """
        Returns: list of (score, meta_dict).
        Score meaning depends on how the index was built:
          - If IndexFlatIP + normalized vectors => cosine similarity in [-1, 1]
          - If L2 index => negative distance is NOT used; FAISS returns L2 distances
        """
        if k <= 0:
            return []

        q_emb = self.embed_query(query)

        scores, idxs = self.index.search(q_emb, k)

        out: List[Tuple[float, Dict]] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            out.append((float(score), self.meta[idx]))
        return out


def format_context(
    hits: List[Tuple[float, Dict]],
    max_chars: int = 4000,
    add_scores: bool = False,
) -> str:
    """
    Converts retrieved chunks into a single context string.

    Changeable parameters:
      - max_chars: caps how much context you inject into the LLM prompt
      - add_scores: debugging toggle
    """
    parts: List[str] = []
    total = 0

    for score, m in hits:
        title = (m.get("title") or "").strip()
        text = (m.get("text") or "").strip()

        if not text:
            continue

        header = f"[{title}]" if title else "[No Title]"
        if add_scores:
            header = f"{header} (score={score:.4f})"

        chunk = f"{header}\n{text}\n"
        if total + len(chunk) > max_chars:
            break

        parts.append(chunk)
        total += len(chunk)

    return "\n---\n".join(parts)


# Optional quick smoke test (not required in your pipeline)
if __name__ == "__main__":
    retriever = DenseRetriever(
        index_path="output/rag_index/faiss.index",
        meta_path="output/rag_index/meta.jsonl",
        embed_model="intfloat/e5-large-v2",
        device="cuda",
        normalize_query=True,
    )

    q = "What is the most famous traditional sport in the US?"
    hits = retriever.search(q, k=5)
    print("Top hits:")
    for s, m in hits[:5]:
        print(f"{s:.4f} | {m.get('title','')}")
    print("\nContext preview:\n")
    print(format_context(hits, max_chars=1200))


