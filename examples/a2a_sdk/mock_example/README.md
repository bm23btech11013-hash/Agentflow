# A2A SDK Integration Example

Demonstrates exposing an agentflow `CompiledGraph` as an A2A-compliant
agent using the official `a2a-sdk`.

## Prerequisites

```bash
pip install agentflow[a2a_sdk]
# or
pip install a2a-sdk
```

## Usage

### 1. Start the server

```bash
python -m examples.a2a_sdk.server
```

This starts an A2A server on `http://localhost:9999` with a simple echo
agent. The agent card is served at
`http://localhost:9999/.well-known/agent-card.json`.

### 2. Call it from the client

In another terminal:

```bash
python -m examples.a2a_sdk.client
```

Expected output:

```
Sending message to A2A agent at http://localhost:9999 ...
Agent response: Echo from agentflow: Hello from the A2A SDK client!
```

## Files

| File | Description |
|------|-------------|
| `server.py` | Builds an agentflow graph, creates an `AgentCard`, starts the A2A server |
| `client.py` | Sends a message to the running server and prints the response |

## How It Works

1. **Server side**: `AgentFlowExecutor` adapts `CompiledGraph.ainvoke()` to
   the `a2a-sdk`'s `AgentExecutor` interface. The SDK handles all HTTP,
   JSON-RPC, and task lifecycle.

2. **Client side**: `delegate_to_a2a_agent()` uses the SDK's `A2AClient` to
   send a message and extract the text response.

## Streaming

To enable streaming, pass `streaming=True`:

```python
create_a2a_server(compiled, card, port=9999, streaming=True)
```

And set `streaming=True` in the agent card:

```python
card = make_agent_card(..., streaming=True)
```
