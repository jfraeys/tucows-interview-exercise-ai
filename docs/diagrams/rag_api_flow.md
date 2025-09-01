# API Flow from User to Response

This diagram illustrates the end-to-end flow of a support ticket from submission to the MCP-compliant JSON response. It shows the interactions between the API endpoint, retrieval module, prompt builder, and the LLM.

```mermaid
flowchart LR
    %% Left branch: user & API
    A["User submits support ticket"] --> B["API Endpoint: /resolve-ticket"]
    B --> C["Prompt Builder (MCP)"]

    %% Right branch: retrieval & LLM
    C --> D["Retrieval Module (FAISS)"]
    C --> E["LLM (cpp-llama) generates response"]
    D --> E

    %% Final output
    E --> F["API returns structured JSON response"]
```

