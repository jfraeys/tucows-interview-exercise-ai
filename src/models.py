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

    def __len__(self):
        return len(self.chunks)

    def __iter__(self):
        return iter(self.chunks)
