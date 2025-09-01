# Document Ingestion & FAISS Pipeline

This diagram shows how raw support documentation is processed and stored in the FAISS vector store. It illustrates the chunking, embedding, and retrieval workflow, as well as how the API interacts with the LLM.

```mermaid
flowchart LR
    subgraph Docs
        RA["Raw Support Docs (.txt, .pdf, .md)"] --> |Chunk & Embed| FA[FAISS Vector Store]
    end
    subgraph API
        U["User Ticket"] --> |POST /resolve-ticket| AP["API Handler"]
        AP --> RB["Prompt Builder"]
        RB --> FS["FAISS Retriever"]
        FS --> RB
        RB --> LL["LLM (cpp-llama)"]
        LL --> AP
        AP --> R["JSON Response"]
    end
```

