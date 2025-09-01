from pydantic import BaseModel


class TicketResponse(BaseModel):
    answer: str
    references: list[str]
    action_required: str


class HealthStatus(BaseModel):
    status: str
    components: dict[str, str]
