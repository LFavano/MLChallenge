from fastapi import FastAPI, Query
from typing import List, Dict
import faiss
import json
import torch
from sentence_transformers import SentenceTransformer
from search_engine import load_index, search_patents

# faiss.randu(1234)  # Set a seed for FAISS randomness control

# Initialize FastAPI app
app = FastAPI(title="Patent Abstract Search API", description="Search multilingual patent abstracts using keywords.")

# Load model and FAISS index
MODEL_NAME = "sentence-transformers/LaBSE"  # Multilingual embedding model
model = SentenceTransformer(MODEL_NAME)
index, metadata = load_index("data/patents.index", "data/metadata.json")


@app.get("/search", summary="Search patent abstracts by keywords")
def search(
        keywords: List[str] = Query(..., description="List of keywords to search"),
        threshold: float = Query(0.5, description="Precision-recall threshold (0-1)")
) -> Dict:
    """Finds the most relevant patent abstracts based on keyword embeddings."""
    # Convert keywords to embeddings
    keyword_embedding = model.encode(" ".join(keywords), convert_to_tensor=True)
    keyword_embedding = keyword_embedding.cpu().numpy().reshape(1, -1)

    # Perform search
    results = search_patents(index, metadata, keyword_embedding, threshold)

    return {"query": keywords, "results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
