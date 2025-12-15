import json
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mega_rag.core.workflow import create_mega_rag_workflow

app = FastAPI(title="MEGA-RAG API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global workflow instance
workflow = None

@app.on_event("startup")
async def startup_event():
    global workflow
    print("Initializing MEGA-RAG workflow...")
    try:
        workflow = create_mega_rag_workflow()
        # Verify index
        if not workflow.retriever.load_indices():
            print("WARNING: No index found. Please run indexing.")
    except Exception as e:
        print(f"Error initializing workflow: {e}")

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    if not workflow:
        return {"error": "Workflow not initialized"}

    async def event_generator():
        try:
            # Iterate through the synchronous generator
            for update in workflow.stream(request.question):
                node = update.get("node")
                state_update = update.get("update", {})
                
                # 1. Send Log (Reasoning)
                trace_msgs = state_update.get("workflow_trace", [])
                if trace_msgs:
                    for msg in trace_msgs:
                        yield json.dumps({"type": "log", "content": msg}) + "\n"
                        # Small delay to make it look nicer on frontend / give event loop time
                        await asyncio.sleep(0.01)
                
                # 2. Check for Final Answer
                if node == "finalize":
                    final_data = {
                        "answer": state_update.get("final_answer"),
                        "is_reliable": state_update.get("is_reliable"),
                    }
                    
                    # Extract sources from the FINAL state (which is spread in state_update)
                    sources = []
                    seen = set()
                    source_metadata = state_update.get("source_metadata", [])
                    for meta in source_metadata:
                        fname = meta.get("filename", "Unknown")
                        if fname not in seen:
                            sources.append(fname)
                            seen.add(fname)
                            
                    final_data["sources"] = sources
                    
                    yield json.dumps({"type": "final", "data": final_data}) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.get("/health")
def health_check():
    return {"status": "ok", "workflow_initialized": workflow is not None}
