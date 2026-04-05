from pydantic import BaseModel

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str

class SourceChunk(BaseModel):
    score: float
    text: str
    source: str
    page: int

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]