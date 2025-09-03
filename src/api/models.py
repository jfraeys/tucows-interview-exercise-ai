from pydantic import BaseModel


class TicketRequest(BaseModel):
    ticket_text: str


class TicketResponse(BaseModel):
    answer: str
    references: list[str]
    action_required: str
