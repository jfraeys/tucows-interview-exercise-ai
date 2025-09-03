from pydantic import BaseModel, Field


class ChunksMetadata(BaseModel):
    source: str
    section: str
    subsection: str | None = None


class Chunks(BaseModel):
    id: str
    text: str
    metadata: ChunksMetadata


class ParserResponse(BaseModel):
    chunks: list[Chunks] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index) -> Chunks:
        return self.chunks[index]

    def to_json(self) -> list[dict]:
        # Serialize to JSON-friendly dicts, skipping None
        return [chunk.model_dump(exclude_none=True) for chunk in self.chunks]


class TicketResponse(BaseModel):
    answer: str
    references: list[str]
    action_required: str
