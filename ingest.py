import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(host="localhost", port=6333, path="./qdrant_data")

COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536

SAMPLE_CHUNKS = [
    "QuantumForge was founded in 2012 by Dr. Elena Rostova and James Sterling in Seattle, Washington.",
    "The flagship product of QuantumForge is the Q-Core Processor, which utilizes rudimentary quantum entanglement to speed up cryptographic hashing.",
    "In 2018, QuantumForge acquired a small robotics startup called MechMinds to expand into automated manufacturing.",
    "QuantumForge's corporate headquarters was relocated to Austin, Texas in 2021 due to favorable tax incentives and a growing tech talent pool.",
    "The company reported a record revenue of $4.2 billion in Q4 of 2023, largely driven by enterprise adoption of their cloud-security infrastructure.",
    "Dr. Elena Rostova stepped down as CEO in January 2024 to focus on research, passing the leadership to former COO, Marcus Thorne."
]

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

def ingest_data():
    """Generates embeddings and upserts data to Qdrant."""
    print("Generating embeddings via OpenAI...")
    
    # Generate embeddings for all chunks in a single batch
    response = openai_client.embeddings.create(
        input=SAMPLE_CHUNKS,
        model=EMBEDDING_MODEL,
    )
    
    points=[]
    