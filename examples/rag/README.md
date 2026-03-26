# RAG Examples (10xScale Agentflow)

This directory provides two progressively more capable Retrieval-Augmented Generation (RAG) examples built with the prebuilt [`RAGAgent`](agentflow/prebuilt/agent/rag.py).

## Files

- `basic_rag.py` – Minimal single-pass RAG: retrieve once, synthesize once, end.
- `advanced_rag.py` – Hybrid pipeline with optional stages: query planning, multiple retrievers, merge, rerank, compress, synthesize.

## When to Use

| Scenario | Use |
|----------|-----|
| Quick demo / smoke test | `basic_rag.py` |
| Show pipeline composition / hybrid retrieval | `advanced_rag.py` |
| Add real vector store (Qdrant / Mem0) | Replace placeholder retrievers with store-backed search |
| Extend with memory | Integrate a `BaseStore` and enrich context prior to synthesis |

## Quick Start

```bash
python examples/rag/basic_rag.py
python examples/rag/advanced_rag.py
```

Environment:

```bash
export OPENAI_API_KEY=your_key
export RAG_MODEL=gpt-4o-mini  # optional
```

## Core Pattern

1. User question enters as messages in `AgentState`
2. One or more retriever nodes enrich context
3. Synthesis node generates answer referencing retrieved snippets
4. Optional loop (disabled in these examples via END condition)

## Replacing the Mock Retriever

Swap the in-memory keyword scorer with a vector store:

```python
# Pseudocode snippet
from agentflow.store import QdrantStore

store = QdrantStore(...)


async def dense_retriever(state: AgentState) -> AgentState:
    query = state.context[-1].text()
    results = await store.asearch({"user_id": "u1"}, query=query, limit=3)
    docs = "\n".join(f"- {r.content}" for r in results)
    state.context.append(Message.text_message(f"[dense]\n{docs}", role="assistant"))
    return state
```

## Advanced Pipeline Slots

| Stage | Purpose | Replace With (Prod) |
|-------|---------|---------------------|
| QUERY_PLAN | Reformulate / decompose | LLM query planner |
| RETRIEVE_n | Candidate gathering | Dense, sparse, metadata, self-query |
| MERGE | Normalize + deduplicate | Score fusion (RRf, weighted) |
| RERANK | Improve ordering | Cross-encoder, LLM judge |
| COMPRESS | Context budget reduction | Map-reduce summarizer |
| SYNTHESIZE | Final answer | LLM w/ structured prompt |

All stages are optional—omit by excluding from `options`.

## Follow-up Loops

Provide a custom `followup_condition(state) -> str` returning either:
- Name of a retriever node (to re-enter retrieval)
- `END` to terminate

## Testing Tips

- Start with `basic_rag.py` to validate tooling / API keys.
- Log `state.context` after each node for debugging.
- Add assertions around presence of `[retrieval]`, `[merge]`, `[compress]` markers.

## Next Steps

- Integrate `QdrantStore` or `Mem0Store` for semantic retrieval
- Add reranker scoring
- Implement conversation memory + adaptive query planning
- Introduce guarded synthesis or citation formatting

Concise, composable, production-ready building blocks—extend responsibly.
