"""
Image similarity search over the pre-built item index.

Uses a CLIP model to turn any photo into a vector, then compares it
(cosine similarity) against every item's vector to find the closest matches.
"""
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "clip-ViT-B-32"

_model = None
_embeddings = None
_metadata = None
_loaded_dir = None


def _load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_index(index_dir):
    global _embeddings, _metadata, _loaded_dir
    emb_path = os.path.join(index_dir, "embeddings.npy")
    meta_path = os.path.join(index_dir, "metadata.pkl")
    _embeddings = np.load(emb_path)
    with open(meta_path, "rb") as f:
        _metadata = pickle.load(f)
    _loaded_dir = index_dir
    return _embeddings, _metadata


def search(image, index_dir, top_k=5):
    """image: a PIL.Image (already RGB). Returns list of dicts, best match first."""
    model = _load_model()
    global _embeddings, _metadata, _loaded_dir
    if _embeddings is None or _loaded_dir != index_dir:
        load_index(index_dir)

    query_emb = model.encode(image, convert_to_numpy=True, normalize_embeddings=True)
    sims = _embeddings @ query_emb
    top_idx = np.argsort(-sims)[:top_k]

    results = []
    for idx in top_idx:
        item = dict(_metadata[idx])
        item["score"] = float(sims[idx])
        results.append(item)
    return results
