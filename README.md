# Patent Abstract Inference Service

This project provides a FastAPI-based inference service that retrieves patent abstracts based on keyword queries. It uses multilingual sentence embeddings and FAISS for efficient search.

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Running the Inference Service

Start the FastAPI server:
```bash
python data_processor.py
python main.py
```
The API will be available at `http://127.0.0.1:8000`.

## 🔎 Using the API

### Search for Patent Abstracts
Send a GET request to:
```bash
http://127.0.0.1:8000/search?keywords=keyword1&keywords=keyword2&threshold=0.5
```
#### Example Response:
```json
{
    "query": ["battery", "efficiency"],
    "results": [
        {
            "abstract": "A technique to enhance battery life in mobile devices.",
            "score": 0.82
        }
    ]
}
```

## ✅ Running Tests
Run unit tests to validate the implementation:
```bash
python -m unittest test_patent_search.py
```
OR using pytest (recommended):
```bash
pytest test_patent_search.py -v
```

## 📁 Project Structure
```
patent_search/
│── main.py                  # FastAPI app
│── search_engine.py         # FAISS search logic
│── data_processor.py        # Embedding & data preprocessing
│── test_patent_search.py    # Unit tests
│── requirements.txt         # Dependencies
│── data/                    # Data storage
│   ├── patents.txt          # Raw abstracts
│   ├── patents.index        # FAISS index
│   ├── metadata.json        # Metadata for abstracts
```

## ⚙️ How It Works
- **Preprocessing (`data_processor.py`)**
  - Reads a dataset of patent abstracts.
  - Converts them into vector embeddings using `sentence-transformers/LaBSE`.
  - Stores embeddings in a FAISS index for fast retrieval.

- **Searching (`search_engine.py`)**
  - Accepts a keyword query, converts it into an embedding.
  - Searches the FAISS index for the most relevant abstracts.
  - Returns results with similarity scores.

- **Inference Service (`main.py`)**
  - Provides a FastAPI interface for querying patents.

### **Pipeline for Integrating New Abstracts**
To ensure the dataset remains up to date, the following method is used:

1. **Retrieve New Patents:**
   - Use APIs to scrape/download the latest patent abstracts (e.g. `patents.google.com` ).

2. **Preprocess and Embed:**
   - Convert new abstracts into embeddings using the existing `SentenceTransformer` model.
   - Store them in a temporary FAISS index.

3. **Merge the New Data:**
   - Append new embeddings to the existing FAISS index without rebuilding from scratch.
   - Create new temporary `metadata.json` to include new abstracts.

4. **Atomic Renames for index swap:**
     - Replace old files with the new temporary onesusing an atomic rename, for metadata and FAISS index

5. **Automate Updates:**
   - Set up a **cron job**  to check for updates periodically.
   - Example: Run the update every 24 hours.

### **Running the Update Manually**
For example if new patents need to be added manually, we can run:
```bash
python update_patents.py
```
which will perform the steps discussed above.