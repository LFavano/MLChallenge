import faiss
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/LaBSE"  # Multilingual embedding model
model = SentenceTransformer(MODEL_NAME)


def process_and_store_data(input_txt: str, index_path: str, metadata_path: str):
    """Processes a text file of patent abstracts and stores them in a FAISS index."""
    with open(input_txt, "r", encoding="utf-8") as f:
        abstracts = [line.strip() for line in f if line.strip()]

    # Encode abstracts into vectors
    embeddings = model.encode(abstracts, convert_to_tensor=True).cpu().numpy()

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save FAISS index and metadata
    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({str(i): abstracts[i] for i in range(len(abstracts))}, f)

    print(f"Processed {len(abstracts)} abstracts and stored them in {index_path}.")


if __name__ == "__main__":
    process_and_store_data("data/patents.txt", "data/patents.index", "data/metadata.json")
