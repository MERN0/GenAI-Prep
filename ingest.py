import os
from dotenv import load_dotenv
from glob import glob
from pypdf import PdfReader
import uuid
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(path="./db/qdrant_data")

COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536
DATA_DIR = "./data"
CHUNK_SIZE = 400      # Characters per chunk
CHUNK_OVERLAP = 50    # Overlap to prevent cutting sentences in half

def setup_collection():
    """Creates the Qdrant collection if it doesn't already exist."""
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"Creating collection '{COLLECTION_NAME}'...")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists. Proceeding to upsert.")


def process_pdfs():
    """Reads PDFs from the data directory and splits them into chunks."""
    os.makedirs(DATA_DIR, exist_ok=True)
    pdf_files = glob(os.path.join(DATA_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"⚠️ No PDFs found in {DATA_DIR}. Please add some and run again.")
        return []

    print(f"Found {len(pdf_files)} PDF(s). Extracting text...")
    chunks = []
    
    for file_path in pdf_files:
        filename = os.path.basename(file_path)
        reader = PdfReader(file_path)
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                # Clean up basic whitespace
                text = " ".join(text.split())
                
                # Slicing the text into overlapping chunks
                start = 0
                while start < len(text):
                    end = start + CHUNK_SIZE
                    chunk_text = text[start:end]
                    chunks.append({
                        "text": chunk_text,
                        "source": filename,
                        "page": page_num + 1
                    })
                    start += (CHUNK_SIZE - CHUNK_OVERLAP)
                    
    return chunks


def ingest_data(chunks):
    """Generates embeddings and upserts data to Qdrant."""    
    if not chunks:
        return
    
    print(f"Generating embeddings for {len(chunks)} chunks via OpenAI...")
    
    # Extract just the text for embedding
    texts_to_embed = [chunk["text"] for chunk in chunks]
    
    # Send to OpenAI in batches (OpenAI handles up to 2048 inputs per batch)
    response = openai_client.embeddings.create(
        input=texts_to_embed,
        model=EMBEDDING_MODEL
    )
    
    points = []
    for i, data in enumerate(response.data):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=data.embedding,
            payload=chunks[i] # This now contains text, source, and page!
        )
        points.append(point)
        
    print(f"Upserting {len(points)} vectors to Qdrant...")
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print("PDF Ingestion complete!")
    
    
if __name__ == "__main__":
    setup_collection()
    pdf_chunks = process_pdfs()
    ingest_data(pdf_chunks)
    qdrant_client.close()
    