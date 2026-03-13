# Pattern 4 — Planner Delegates with Cross-Boundary INPUT_REQUIRED

## What this demonstrates

**INPUT_REQUIRED propagating ACROSS agent boundaries.**

Unlike Pattern 2 (where the orchestrator gathers info itself before delegating), here the **specialist agent** (CurrencyAgent) decides it needs more info and returns `INPUT_REQUIRED`. The PlannerAgent **relays** that question back to the human, then forwards the human's answer back to CurrencyAgent on the same task.

## Architecture

```
User
 ↕  (A2A, port 10001)
PlannerAgent
 ↕  (A2A, port 10000)
CurrencyAgent
 ↕  (Frankfurter API)
Real-time exchange rates
```

## Message flow

```
Step 1: User → "convert currency"
Step 2: PlannerAgent classifies → currency → delegates to CurrencyAgent
Step 3: CurrencyAgent → INPUT_REQUIRED "What amount and which currencies?"
Step 4: PlannerAgent receives INPUT_REQUIRED → relays to User (INPUT_REQUIRED)
Step 5: User → "100 USD to EUR"
Step 6: PlannerAgent forwards to CurrencyAgent (SAME task_id)
Step 7: CurrencyAgent calls Frankfurter API → COMPLETED "100 USD = 93.05 EUR"
Step 8: PlannerAgent receives COMPLETED → returns result to User
```

**Key:** Two separate A2A task conversations are maintained:
- **User ↔ PlannerAgent** — tracked by PlannerAgent's task store
- **PlannerAgent ↔ CurrencyAgent** — tracked in PlannerAgent's `_sessions` dict

## How to run

```bash
# Terminal 1 — Start CurrencyAgent
python -m examples.a2a_sdk.pattern4_planner_delegates.currency_agent

# Terminal 2 — Start PlannerAgent  
python -m examples.a2a_sdk.pattern4_planner_delegates.planner_agent

# Terminal 3 — Run the client
python -m examples.a2a_sdk.pattern4_planner_delegates.client
```

## Example session

```
You: convert currency

Agent asks: I can help with currency conversion! Please provide the
amount, the source currency, and the target currency (e.g., 100 USD to EUR).
  (The agent needs more info — type your answer)

You (follow-up): 100 USD to INR

Agent: Based on the current exchange rate, 100 USD is approximately
8,362.50 INR.
```

## Key difference from Pattern 2

| Aspect | Pattern 2 (Orchestrator) | Pattern 4 (Planner Delegates) |
|--------|------------------------|------------------------------|
| Who gathers info? | Orchestrator itself | Specialist (CurrencyAgent) |
| INPUT_REQUIRED crosses boundary? | No | **Yes** |
| Multi-turn with specialist? | No (single delegation) | **Yes** (same task_id) |
| Session tracking | None needed | PlannerAgent tracks remote task_ids |

## Files

| File | Description |
|------|-----------|
| `currency_agent.py` | CurrencyAgent on port 10000 with INPUT_REQUIRED support |
| `planner_agent.py` | PlannerAgent on port 10001 — delegates and relays INPUT_REQUIRED |
| `client.py` | Interactive CLI client for PlannerAgent |
