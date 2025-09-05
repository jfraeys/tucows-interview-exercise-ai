"""
FastAPI endpoint wrapping the existing RetrievalQA RAG + MCP system.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

import src.api.models as models
from src.api import web_logger
from src.core import RetrievalQA, initialize_rag

logger = web_logger.get_web_logger("logs/api.log")
uvicorn_logger = web_logger.get_uvicorn_logger()

INDEX_PATH = "data/faiss_index/faiss.index"
METADATA_PATH = "data/faiss_index/metadata.json"
MODEL_PATH = "models/Phi-3-mini-4k-instruct-fp16.gguf"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize RAG pipeline
    vectorstore, llm = initialize_rag(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
    )

    app.state.rag = RetrievalQA(vectorstore=vectorstore, llm=llm)

    # Warm up the LLM once before serving
    await app.state.rag.warmup()
    uvicorn_logger.info("RAG system initialized and warmed up.")

    yield  # API runs here

    # closing connections, releasing resources
    del app.state.rag


app = FastAPI(lifespan=lifespan)


@app.post("/resolve-ticket", response_model=models.TicketResponse)
async def resolve_ticket(ticket: models.TicketRequest) -> models.TicketResponse:
    """
    Resolve a support ticket using RetrievalQA.
    Only accepts JSON input matching {"ticket_text": "message"}.
    """
    ticket_text = ticket.ticket_text.strip()
    if not ticket_text:
        raise HTTPException(status_code=400, detail="Empty ticket text")

    # Initialize components
    vectorstore, llm = initialize_rag(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
    )
    rag_system = RetrievalQA(vectorstore=vectorstore, llm=llm, top_k=5)

    try:
        # Generate answer
        answer = await rag_system.answer(ticket_text)

        return models.TicketResponse(
            answer=answer["answer"],
            references=answer.get("references", []),
            action_required=answer["action_required"],
        )
    except Exception as e:
        logger.exception(f"Failed to resolve ticket: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Entry point for the knowledge-assistant CLI command."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
