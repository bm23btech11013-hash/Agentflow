# Pattern 2 — Orchestrator Delegates to Specialists

```
Human
    │
    ▼
OrchestratorAgent (has LLM, reasons about intent)
    │
    ├──► CurrencyAgent  ("100 USD to INR" — complete info)
    │         │ COMPLETED
    │         ▼
    │    "8,362 INR"
    │
    ├──► WeatherAgent  ("London" — complete info)
    │         │ COMPLETED
    │         ▼
    │    "Rainy, 15°C"
    │
    ▼
Human ← orchestrated final answer
```

## When to use

One entry point, multiple specialist agents behind it.

- **INPUT_REQUIRED**: Orchestrator clarifies with human **BEFORE** delegating. Specialists always get complete info.
- Different from API: Specialists are autonomous — they have their own LLM, can handle complex reasoning, return rich artifacts. An API just returns raw data.

## Files

| File | Port | Description |
|------|------|-------------|
| `currency_agent.py` | 10000 | CurrencyAgent specialist (Frankfurter API) |
| `weather_agent.py` | 10001 | WeatherAgent specialist (Open-Meteo API) |
| `orchestrator.py` | 10002 | OrchestratorAgent — routes & gathers info |
| `client.py` | — | Interactive CLI client |

## Running

```bash
# Terminal 1 — start CurrencyAgent specialist
python -m examples.a2a_sdk.pattern2_orchestrator.currency_agent

# Terminal 2 — start WeatherAgent specialist
python -m examples.a2a_sdk.pattern2_orchestrator.weather_agent

# Terminal 3 — start OrchestratorAgent
python -m examples.a2a_sdk.pattern2_orchestrator.orchestrator

# Terminal 4 — start the client
python -m examples.a2a_sdk.pattern2_orchestrator.client
```

## Try these conversations

1. **Currency** (complete info → direct delegation):
   ```
   You: 100 USD to INR
   Agent: 100 USD = 8,362 INR (as of 2025-03-07)
   ```

2. **Weather** (complete info → direct delegation):
   ```
   You: weather in London
   Agent: London: Rainy, 15°C, wind 12 km/h
   ```

3. **Currency** (incomplete → orchestrator gathers first):
   ```
   You: convert currency
   Agent: What amount would you like to convert? (INPUT_REQUIRED)
   You: 50 EUR to GBP
   Agent: 50 EUR = 42.15 GBP
   ```

4. **General** (answered by orchestrator directly):
   ```
   You: what is the capital of France?
   Agent: The capital of France is Paris.
   ```
