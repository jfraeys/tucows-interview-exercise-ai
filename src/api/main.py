from fastapi import Body, FastAPI

import src.api.models as models

app = FastAPI()


# Dummy lightweight FAISS check
def check_faiss_index():
    """Check if FAISS index is loadable without full vectors"""
    try:
        # Attempt to load a minimal index or metadata only
        from src.data_handling.faiss_index import load_index_metadata

        metadata = load_index_metadata()  # just a small object, not full vectors
        if metadata:
            return "ok"
        return "missing"
    except Exception as e:
        return f"error: {e}"


# Dummy lightweight LLM check
def check_llm():
    """Simple prompt to verify LLM is responsive"""

    try:
        from src.core.llm_wrapper import LLM

        llm = LLM(load_full_model=False)  # only load minimal model for ping
        test_output = llm.generate("ping", max_tokens=5)
        return "ok" if test_output else "failed"
    except Exception as e:
        return f"error: {e}"


@app.get("/health", response_model=models.HealthStatus)
async def health_check():
    """
    Health check endpoint to verify system components.
    Checks FAISS index and LLM availability.
    """

    components = {
        "faiss": check_faiss_index(),
        "llm": check_llm(),
        # Add other components here if needed
    }
    overall_status = "ok" if all(v == "ok" for v in components.values()) else "degraded"
    return models.HealthStatus(status=overall_status, components=components)


@app.post("/resolve-ticket", response_model=models.TicketResponse)
async def resolve_ticket(ticket_text: str = Body(..., media_type="text/plain")):
    """
    RAG + MCP pipeline:
    1. Retrieve relevant document chunks from FAISS.
    2. Construct MCP prompt.
    3. Generate response with LLM.
    4. Return MCP-compliant JSON.
    """
    # Dummy implementation
    return models.TicketResponse(
        answer=f"Resolved: {ticket_text}",
        references=["Policy: Example Doc, Section 1.2"],
        action_required="escalate_to_support",
    )
