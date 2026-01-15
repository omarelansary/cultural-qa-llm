import os
import json
import argparse
from typing import List, Dict

import numpy as np
from tqdm import tqdm

import torch
from sentence_transformers import SentenceTransformer
import faiss


def read_jsonl(path: str) -> List[Dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_jsonl", required=True, help="Path to wiki_chunks.jsonl")
    ap.add_argument("--out_dir", default="rag_index", help="Output dir for index + meta")
    ap.add_argument("--model_name", default="intfloat/e5-large-v2", help="SentenceTransformer model")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_docs", type=int, default=-1, help="For debugging; -1 means all")
    ap.add_argument("--use_gpu", action="store_true", help="Use GPU for embeddings if available")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    index_path = os.path.join(args.out_dir, "faiss.index")
    meta_path = os.path.join(args.out_dir, "meta.jsonl")

    print(f"Loading chunks: {args.chunks_jsonl}")
    docs = read_jsonl(args.chunks_jsonl)
    if args.max_docs > 0:
        docs = docs[: args.max_docs]
    print(f"Docs: {len(docs):,}")

    device = "cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"
    print(f"Loading embedding model: {args.model_name} on {device}")
    model = SentenceTransformer(args.model_name, device=device)

    # E5 expects prefixes
    passages = []
    for d in docs:
        title = (d.get("title") or "").strip()
        text = (d.get("text") or "").strip()
        if title:
            passages.append(f"passage: {title}\n{text}")
        else:
            passages.append(f"passage: {text}")

    # Embed in batches
    all_embs = []
    for i in tqdm(range(0, len(passages), args.batch_size), desc="Embedding passages"):
        batch = passages[i : i + args.batch_size]
        emb = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        all_embs.append(emb.astype("float32"))

    embs = np.vstack(all_embs)  # (N, D)
    embs = l2_normalize(embs)   # cosine via inner product
    N, D = embs.shape
    print(f"Embeddings: {N:,} x {D}")

    # FAISS index for cosine: IndexFlatIP (or IVF if you get huge)
    index = faiss.IndexFlatIP(D)
    index.add(embs)
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index: {index_path}")

    # Write meta in the SAME ORDER as embeddings
    with open(meta_path, "w", encoding="utf-8") as f:
        for d in docs:
            out = {
                "id": d.get("id"),
                "title": d.get("title"),
                "text": d.get("text"),
                "source": d.get("source"),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
