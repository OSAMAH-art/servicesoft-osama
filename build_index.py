"""
Builds the searchable image index from the StorageList Excel export.

For every row that has a PictureLocation, this:
  1. downloads the image
  2. computes a CLIP embedding (a numeric "fingerprint" of what's in the photo)
  3. saves all embeddings + the item's details (code, name, qty, shelf...) to disk

Run this ONCE up front (and again whenever you want to refresh the catalog).
It needs internet access to reach the image URLs, and it can take a while for
~20,000 images — that's expected. It saves progress every 50 items, and can be
safely stopped and re-run later (--resume, on by default) to continue where it
left off.

Usage:
    python build_index.py --excel data/StorageList.xlsx --out data/index

    # quick test run on the first 200 items only:
    python build_index.py --excel data/StorageList.xlsx --out data/index --limit 200
"""
import argparse
import os
import pickle
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from PIL import Image
from sentence_transformers import SentenceTransformer

MODEL_NAME = "clip-ViT-B-32"

# Columns from the export we keep alongside each embedding, so search results
# can show the item's real details. Adjust this list if your export differs.
KEEP_COLUMNS = [
    "ItemID",
    "ItemNumber",
    "ItemName",
    "Quantity",
    "ShelfNo",
    "StorageName",
    "BranchName",
    "Description",
    "Devices",
    "PictureLocation",
]


def download_image(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def build_index(excel_path, out_dir, limit=None, resume=True, save_every=50):
    os.makedirs(out_dir, exist_ok=True)
    embeddings_path = os.path.join(out_dir, "embeddings.npy")
    meta_path = os.path.join(out_dir, "metadata.pkl")

    df = pd.read_excel(excel_path)
    df = df[df["PictureLocation"].notna()].reset_index(drop=True)
    if limit:
        df = df.head(limit)

    print(f"Loading CLIP model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    embeddings = []
    metadata = []
    processed_ids = set()

    if resume and os.path.exists(embeddings_path) and os.path.exists(meta_path):
        embeddings = list(np.load(embeddings_path))
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        processed_ids = {m["ItemID"] for m in metadata}
        print(f"Resuming: {len(processed_ids)} items already indexed, skipping those.")

    total = len(df)
    failed = 0
    for i, row in df.iterrows():
        item_id = row["ItemID"]
        if item_id in processed_ids:
            continue

        img = download_image(row["PictureLocation"])
        if img is None:
            failed += 1
            continue

        emb = model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(emb)
        metadata.append({col: row.get(col) for col in KEEP_COLUMNS})

        done = len(metadata)
        if done % save_every == 0:
            print(f"[{i + 1}/{total}] indexed so far: {done}  (failed downloads: {failed})")
            np.save(embeddings_path, np.array(embeddings))
            with open(meta_path, "wb") as f:
                pickle.dump(metadata, f)

    np.save(embeddings_path, np.array(embeddings))
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Done. {len(embeddings)} items indexed, {failed} images failed to download.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", required=True, help="Path to the StorageList .xlsx export")
    parser.add_argument("--out", default="data/index", help="Where to save the index files")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (for testing)")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing progress and start fresh")
    args = parser.parse_args()

    build_index(args.excel, args.out, limit=args.limit, resume=not args.no_resume)
