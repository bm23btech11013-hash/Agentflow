# Pattern 3 — Smart Client Routes Directly

```
Smart Client (has discovery + routing logic)
    │
    ├──► CurrencyAgent (port 10000) — for money queries
    ├──► WeatherAgent  (port 10001) — for weather queries
    └──► GeneralAgent  (port 10002) — for everything else
```

## When to use

Client discovers agents via **Registry/AgentCard** and routes directly — no orchestrator.

- Client fetches `/.well-known/agent-card.json` from each agent
- Routes by matching query keywords against agent tags and skills
- No LLM overhead for routing — pure keyword matching
- Agents are independently deployable and discoverable

## Key difference from Pattern 2

| | Pattern 2 (Orchestrator) | Pattern 3 (Smart Client) |
|---|---|---|
| **Router** | Orchestrator agent (has LLM) | Client code (keyword matching) |
| **LLM cost for routing** | Yes (classification LLM call) | No (tag matching) |
| **Info gathering** | Orchestrator gathers via INPUT_REQUIRED | Client sends as-is |
| **Single entry point** | Yes (orchestrator URL) | No (client picks agent URL) |
| **Complexity** | Higher (extra agent to deploy) | Lower (logic in client) |

## Files

| File | Port | Description |
|------|------|-------------|
| `currency_agent.py` | 10000 | CurrencyAgent specialist |
| `weather_agent.py` | 10001 | WeatherAgent specialist |
| `general_agent.py` | 10002 | GeneralAgent (catch-all) |
| `client.py` | — | Smart Client with discovery + routing |

## Running

```bash
# Terminal 1 — CurrencyAgent
python -m examples.a2a_sdk.pattern3_smart_client.currency_agent

# Terminal 2 — WeatherAgent
python -m examples.a2a_sdk.pattern3_smart_client.weather_agent

# Terminal 3 — GeneralAgent
python -m examples.a2a_sdk.pattern3_smart_client.general_agent

# Terminal 4 — Smart Client
python -m examples.a2a_sdk.pattern3_smart_client.client
```

## Discovery output

```
Discovering agents...
  Discovered: CurrencyAgent at http://localhost:10000
    Tags: ['currency', 'finance', 'exchange-rate', 'money']
  Discovered: WeatherAgent at http://localhost:10001
    Tags: ['weather', 'forecast', 'temperature', 'climate']
  Discovered: GeneralAgent at http://localhost:10002
    Tags: ['general', 'knowledge', 'qa', 'assistant']

3 agent(s) discovered. Routing by AgentCard tags.
```

## Example session

```
You: 100 USD to EUR
  [Routing to CurrencyAgent at http://localhost:10000]
Agent (CurrencyAgent): 100 USD = 92.15 EUR (as of 2025-03-07)

You: weather in Tokyo
  [Routing to WeatherAgent at http://localhost:10001]
Agent (WeatherAgent): Tokyo: Partly cloudy, 18°C, wind 8 km/h

You: who wrote Hamlet?
  [Routing to GeneralAgent at http://localhost:10002]
Agent (GeneralAgent): Hamlet was written by William Shakespeare.
```
