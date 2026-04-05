import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

# import schemas as sc
from schemas import *

# Load environment variables
load_dotenv()

# Global Client Variables
COLLECTION_NAME = "knowledge_base"

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
    
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = QdrantClient(path="./db/qdrant_data")

# Ensure collection exists before serving traffic
if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
    print("Warning: Qdrant collection does not exist. Please run ingest.py first.")
    
# yield
# Cleanup on shutdown (if necessary)
# qdrant_client.close()

app = FastAPI(title="RAG API",)
#  lifespan=lifspan

@app.post('/query', response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    try:
        # 1. Embed the user's question
        embed_response = openai_client.embeddings.create(
            input=request.question,
            model="text-embedding-3-small"
        )
        
        query_vector = embed_response.data[0].embedding
        
        # 2. Query Qdrant for the top 5 closest chunks
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=5
        )
        

        sources = []
        context_texts = []

        # 3. Extract text payloads and format sources as list of dicts
        for matched_res in search_results.points:
            text = matched_res.payload.get('text', '')
            source_file = matched_res.payload.get("source", "Unknown PDF")
            page_num = matched_res.payload.get("page", 0)
            # sources.append({"score": matched_res.score, "text": text})
            
            sources.append(SourceChunk(
                score=matched_res.score, 
                text=text, 
                source=source_file, 
                page=page_num
            ))
            
            
            context_texts.append(f"[{source_file} - Page {page_num}]: {text}")

        # 4. Construct Prompt for GPT-4o-mini
        context_str = "\n\n---\n\n".join(context_texts)

        system_prompt = (
            "You are a highly accurate and helpful assistant. "
            "You must answer the user's question using ONLY the provided context chunks below. "
            "If the provided context does not contain the answer, you must reply strictly with: 'I don't know'."
        )

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {request.question}"
        
        # 5. Call OpenAI to generate the final answer
        chat_response = openai_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # temperature=0 # model supports only default temp "0"
        )
        
        answer = chat_response.choices[0].message.content
        
        # qdrant_client.close()
        # 6. Return standard response
        return QueryResponse(answer=answer, sources=sources)
        
    except Exception as e:
        # Catch errors (e.g., OpenAI API rate limits, Qdrant disconnects) and return a 500
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")