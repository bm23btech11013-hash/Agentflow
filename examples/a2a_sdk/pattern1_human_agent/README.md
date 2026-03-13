# Pattern 1 — Human ↔ Agent (Direct)

```
Human Client
    │  "convert 100 USD to INR"
    ▼
CurrencyAgent
    │  INPUT_REQUIRED "which date?"
    ▼
Human Client
    │  "today"
    ▼
CurrencyAgent
    │  COMPLETED "100 USD = 8,362 INR"
    ▼
Human Client
```

## When to use

End user talking **directly** to a specialist agent.

- **INPUT_REQUIRED** works perfectly — the human answers directly.
- Different from an API: an API can't ask follow-up questions. An agent **can**.

## Files

| File | Description |
|------|-------------|
| `agent.py` | CurrencyAgent server (port 10000) with INPUT_REQUIRED support |
| `client.py` | Interactive CLI client that handles multi-turn conversations |

## Running

```bash
# Terminal 1 — start the agent
python -m examples.a2a_sdk.pattern1_human_agent.agent

# Terminal 2 — start the client
python -m examples.a2a_sdk.pattern1_human_agent.client
```

## Try these conversations

1. **Complete info** → direct answer:
   ```
   You: 100 USD to INR
   Agent: 100 USD = 8,362 INR (as of 2025-03-07)
   ```

2. **Missing info** → agent asks follow-up:
   ```
   You: convert currency
   Agent: What amount would you like to convert? (INPUT_REQUIRED)
   You: 50 EUR
   Agent: Which currency would you like to convert to? (INPUT_REQUIRED)
   You: GBP
   Agent: 50 EUR = 42.15 GBP (as of 2025-03-07)
   ```
