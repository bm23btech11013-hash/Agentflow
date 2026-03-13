# Currency Agent — agentflow + A2A Protocol

Mirrors the official [a2a-samples](https://github.com/a2aproject/a2a-samples)
LangGraph currency agent, built with agentflow instead.

## Prerequisites

```bash
pip install agentflow[a2a_sdk,litellm]
```

Set your LLM API key in `.env` at the project root:

```
GEMINI_API_KEY=your-key-here
```

The Frankfurter exchange-rate API is free and requires no key.

## Run

```bash
# Terminal 1 — start the agent server
python -m examples.a2a_sdk.currency.server

# Terminal 2 — query it
python -m examples.a2a_sdk.currency.client
```

You can also run the agent graph standalone (no A2A server):

```bash
python -m examples.a2a_sdk.currency.agent
```

## What It Does

1. Receives natural-language currency queries via the A2A protocol.
2. The LLM decides to call the `get_exchange_rate` tool with the right
   currency codes and amount.
3. The tool hits the [Frankfurter API](https://api.frankfurter.app) for
   live rates (free, no API key).
4. The LLM formats the result into a human-readable answer.
5. The answer is returned to the A2A client.

## Architecture

```
A2A Client  ──HTTP/JSON-RPC──▶  A2A Server (a2a-sdk)
                                      │
                                AgentFlowExecutor
                                      │
                              CompiledGraph (ReAct)
                              ┌───────┴───────┐
                           LLM Node      ToolNode
                          (Gemini)    (get_exchange_rate)
                                           │
                                   Frankfurter API
```

## Files

| File | Description |
|------|-------------|
| `agent.py` | Agentflow graph: LLM node + ToolNode in a ReAct loop |
| `server.py` | Wraps the graph as an A2A server on port 10000 |
| `client.py` | Sends sample currency queries and prints responses |
