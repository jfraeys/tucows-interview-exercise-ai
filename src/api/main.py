"""
FastAPI endpoint wrapping the existing RetrievalQA RAG + MCP system.
"""

from fastapi import FastAPI, HTTPException

import src.api.models as models
from src.core import llm_wrapper, rag_pipeline, retrieval
from src.utils import logging_helper

logger = logging_helper.get_logger(__name__)

app = FastAPI()

INDEX_FILE = "data/faiss_index/faiss.index"
DATA_FILE = "data/faiss_index/metadata.json"
MODEL_PATH = "models/Phi-3-mini-4k-instruct-fp16.gguf"


def _initialize_components(
    model_path: str = MODEL_PATH,
    model_repo: str | None = None,
    model_filename: str | None = None,
    n_gpu_layers: int = 0,
):
    """
    Initialize the vector store and LLM wrapper.
    """
    vectorstore = retrieval.VectorStore(
        index_path=INDEX_FILE,
        metadata_path=DATA_FILE,
    )

    if model_path:
        llm = llm_wrapper.LLMWrapper(model_path=model_path, n_gpu_layers=n_gpu_layers)
    else:
        llm = llm_wrapper.LLMWrapper(
            repo_id=model_repo,
            filename=model_filename,
            n_gpu_layers=n_gpu_layers,
        )

    return vectorstore, llm


@app.post("/resolve-ticket", response_model=models.TicketResponse)
async def resolve_ticket(ticket: models.TicketRequest):
    """
    Resolve a support ticket using RetrievalQA.
    Only accepts JSON input matching {"ticket_text": "message"}.
    """
    ticket_text = ticket.ticket_text.strip()
    if not ticket_text:
        raise HTTPException(status_code=400, detail="Empty ticket text")

    # Initialize components
    vectorstore, llm = _initialize_components(n_gpu_layers=-1)
    rag_system = rag_pipeline.RetrievalQA(vectorstore=vectorstore, llm=llm, top_k=7)

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
