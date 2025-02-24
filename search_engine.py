import faiss
import json
import numpy as np

def load_index(index_path: str, metadata_path: str):
    """Loads the FAISS index and metadata."""
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

def search_patents(index, metadata, query_vector, threshold=0.5, top_k=10):
    """Searches for the most relevant patent abstracts given a query vector."""
    distances, indices = index.search(query_vector, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if dist < threshold:
            break
        results.append({
            "abstract": metadata[str(idx)],
            "score": float(dist)
        })
    return results
