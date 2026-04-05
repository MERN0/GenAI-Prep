# 📚 RAG System with PDF Corpus

An end-to-end Retrieval-Augmented Generation (RAG) application built with FastAPI, Qdrant (Local Vector DB), OpenAI, and Streamlit.

This system allows you to ingest a local directory of PDF documents, generate vector embeddings, and ask natural language questions. The AI will answer using *only* the ingested corpus and will cite its sources (including filename and page number).

## 🛠 Tech Stack

* **Backend:** FastAPI, Uvicorn
* **Frontend:** Streamlit
* **Vector Database:** Qdrant (Local Mode)
* **AI/LLM:** OpenAI (`gpt-4o-mini` for generation, `text-embedding-3-small` for embeddings)
* **Document Processing:** PyPDF

## 📂 Project Structure

```text
├── data/                  # Drop your PDF files here
├── qdrant_data/           # Local vector database storage (auto-generated)
├── .env                   # Environment variables (API keys)
├── .gitignore             # Git ignore file
├── ingest.py              # Script to extract, chunk, and embed PDFs
├── main.py                # FastAPI backend server
├── requirements.txt       # Python dependencies
└── streamlit_app.py       # Streamlit frontend UI
```

## 🏃‍♂️ Running the Application

## Setup Instructions

### 1. Create and Activate a Virtual Environment

Keep your dependencies isolated by creating a virtual environment:

**Bash**

```
python -m venv venv

# On Mac/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

Install all required libraries, including FastAPI, Qdrant, OpenAI, PyPDF, and Streamlit.

**Bash**

```
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory (or copy `.env.example` if you have one) and add your OpenAI API key:

**Code snippet**

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Ingest Your PDF Data

1. Create a folder named `data` in the root directory.
2. Drop one or more `.pdf` files into the `data` folder.
3. Run the ingestion script to extract the text, split it into chunks, generate embeddings, and save them to Qdrant:

**Bash**

```
python ingest.py
```

*(You should see terminal output confirming the number of pages processed and chunks upserted.)*

To run the full stack, you need to start both the FastAPI backend and the Streamlit frontend in separate terminal windows.

### Step 1: Start the Backend API

In your first terminal (with your virtual environment activated), start the FastAPI server:

**Bash**

```
uvicorn main:app --reload
```

*The API is now running at `http://localhost:8000`.*
*Interactive API documentation is available at `http://localhost:8000/docs`.*

### Step 2: Start the Frontend UI

Open a  **new terminal tab** , activate your virtual environment again, and start the Streamlit app:

**Bash**

```
streamlit run streamlit_app.py
```

*The UI will automatically open in your web browser at `http://localhost:8501`.*

---

## 🧪 Testing via CLI (Optional)

If you prefer to test the backend API directly without the UI, you can use `curl`:

**Bash**

```
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the main topic of the uploaded document?"}'
```

**Expected JSON Response:**

**JSON**

```
{
  "answer": "The main topic is...",
  "sources": [
    {
      "score": 0.8543,
      "text": "Extracted text snippet here...",
      "source": "document_name.pdf",
      "page": 4
    }
  ]
}
```
