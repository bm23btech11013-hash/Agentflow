"""
Pattern 3 — Smart Client that routes directly to agents.

The client itself discovers agents via their AgentCards (the standard
A2A discovery mechanism at ``/.well-known/agent-card.json``), inspects
their skills/tags, and routes each user query to the best agent.

There is NO orchestrator agent — the client IS the router.

Flow::

    Smart Client (has discovery + routing logic)
        │
        ├──► CurrencyAgent (port 10000) — for money queries
        ├──► WeatherAgent  (port 10001) — for weather queries
        └──► GeneralAgent  (port 10002) — for everything else

Key difference from Pattern 2:
    - Pattern 2: Orchestrator (another agent) does routing with an LLM
    - Pattern 3: Client does routing with keyword matching on AgentCards
    - No LLM overhead for routing — just tag/skill matching
    - Client discovers agents at startup, can refresh dynamically

Usage::

    # Start all three agents first:
    #   python -m examples.a2a_sdk.pattern3_smart_client.currency_agent
    #   python -m examples.a2a_sdk.pattern3_smart_client.weather_agent
    #   python -m examples.a2a_sdk.pattern3_smart_client.general_agent

    python -m examples.a2a_sdk.pattern3_smart_client.client
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field

import httpx
from a2a.client import A2AClient
from a2a.types import (
    AgentCard,
    Message as A2AMessage,
    MessageSendParams,
    Role,
    SendMessageRequest,
    TextPart,
)

# ------------------------------------------------------------------ #
#  Agent Registry — the client discovers agents at startup             #
# ------------------------------------------------------------------ #

# Known agent endpoints — in production these would come from a
# registry service, DNS-SD, or manual configuration.
AGENT_ENDPOINTS = [
    "http://localhost:10000",  # CurrencyAgent
    "http://localhost:10001",  # WeatherAgent
    "http://localhost:10002",  # GeneralAgent
]


@dataclass
class DiscoveredAgent:
    """An agent discovered via its AgentCard."""

    name: str
    description: str
    url: str
    tags: list[str] = field(default_factory=list)
    skill_keywords: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ #
#  Discovery — fetch AgentCards from well-known endpoints               #
# ------------------------------------------------------------------ #


async def discover_agents(
    endpoints: list[str],
    timeout: float = 10.0,
) -> list[DiscoveredAgent]:
    """Discover agents by fetching their AgentCards.

    Each A2A agent exposes ``/.well-known/agent-card.json`` which
    contains name, description, skills, and tags.  The smart client
    reads these to decide where to route queries.
    """
    agents: list[DiscoveredAgent] = []

    async with httpx.AsyncClient(timeout=timeout) as http:
        for endpoint in endpoints:
            try:
                url = f"{endpoint}/.well-known/agent-card.json"
                resp = await http.get(url)
                resp.raise_for_status()
                card_data = resp.json()

                # Extract tags from all skills
                tags: list[str] = []
                keywords: list[str] = []
                for skill in card_data.get("skills", []):
                    tags.extend(skill.get("tags", []))
                    keywords.append(skill.get("name", "").lower())
                    keywords.append(skill.get("description", "").lower())
                    keywords.extend(
                        ex.lower() for ex in skill.get("examples", [])
                    )

                agent = DiscoveredAgent(
                    name=card_data.get("name", "Unknown"),
                    description=card_data.get("description", ""),
                    url=endpoint,
                    tags=[t.lower() for t in tags],
                    skill_keywords=keywords,
                )
                agents.append(agent)
                print(f"  Discovered: {agent.name} at {agent.url}")
                print(f"    Tags: {agent.tags}")

            except Exception as exc:
                print(f"  Could not discover agent at {endpoint}: {exc}")

    return agents


# ------------------------------------------------------------------ #
#  Routing — keyword/tag matching (no LLM needed)                      #
# ------------------------------------------------------------------ #

# Keywords that map to specific agent types
CURRENCY_KEYWORDS = {
    "currency", "convert", "exchange", "rate", "usd", "eur", "gbp",
    "jpy", "inr", "cad", "aud", "money", "forex", "dollar", "euro",
    "pound", "yen", "rupee",
}
WEATHER_KEYWORDS = {
    "weather", "temperature", "forecast", "rain", "snow", "sunny",
    "cloudy", "wind", "climate", "celsius", "fahrenheit", "storm",
    "humidity",
}


def route_query(
    query: str,
    agents: list[DiscoveredAgent],
) -> DiscoveredAgent | None:
    """Route a query to the best agent using keyword matching.

    Strategy:
    1. Check agent tags and skill keywords against the query
    2. Score each agent by how many relevant keywords match
    3. Return the highest-scoring agent (or the "general" fallback)

    No LLM overhead — pure keyword/tag matching against AgentCards.
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())

    best_agent: DiscoveredAgent | None = None
    best_score = 0
    general_agent: DiscoveredAgent | None = None

    for agent in agents:
        # Check if this is the general/fallback agent
        if "general" in agent.tags or "qa" in agent.tags:
            general_agent = agent

        score = 0

        # Score based on tag matches
        for tag in agent.tags:
            if tag in query_lower:
                score += 3  # Tag in query text: strong signal
            if tag in query_words:
                score += 2  # Tag matches a word exactly

        # Score based on keyword matches from skills
        for keyword in agent.skill_keywords:
            for word in query_words:
                if word in keyword or keyword in query_lower:
                    score += 1

        # Check against our known keyword sets for extra confidence
        if any(kw in query_lower for kw in CURRENCY_KEYWORDS):
            if "currency" in agent.tags or "finance" in agent.tags:
                score += 5

        if any(kw in query_lower for kw in WEATHER_KEYWORDS):
            if "weather" in agent.tags or "forecast" in agent.tags:
                score += 5

        if score > best_score:
            best_score = score
            best_agent = agent

    # If no strong match, fall back to general agent
    if best_score < 2 and general_agent:
        return general_agent

    return best_agent or general_agent


# ------------------------------------------------------------------ #
#  Text extraction                                                     #
# ------------------------------------------------------------------ #


def _extract_response_text(payload) -> str:
    """Extract text from a completed task."""
    try:
        if hasattr(payload, "artifacts") and payload.artifacts:
            texts: list[str] = []
            for artifact in payload.artifacts:
                for p in artifact.parts:
                    if hasattr(p, "root") and hasattr(p.root, "text") and p.root.text:
                        texts.append(p.root.text)
                    elif hasattr(p, "text") and isinstance(p.text, str) and p.text:
                        texts.append(p.text)
            if texts:
                return "\n".join(texts)

        if hasattr(payload, "status") and payload.status and payload.status.message:
            for p in payload.status.message.parts:
                if hasattr(p, "root") and hasattr(p.root, "text") and p.root.text:
                    return p.root.text
                if hasattr(p, "text") and isinstance(p.text, str) and p.text:
                    return p.text
    except Exception:
        pass
    return "No response."


# ------------------------------------------------------------------ #
#  Chat loop                                                           #
# ------------------------------------------------------------------ #


async def chat() -> None:
    """Interactive CLI with smart routing.

    1. Discover all agents at startup via AgentCards
    2. For each user query, route to the best agent
    3. Send the query directly — no orchestrator middleman

    Uses a persistent ``context_id`` per agent so each specialist
    retains conversation history across multiple tasks.
    """
    print("=" * 60)
    print("  Pattern 3 — Smart Client Routes Directly")
    print("=" * 60)
    print()
    print("  Discovering agents...")
    print()

    agents = await discover_agents(AGENT_ENDPOINTS)

    if not agents:
        print("  No agents discovered! Make sure they are running.")
        print("  Start agents on ports 10000, 10001, 10002.")
        return

    print()
    print(f"  {len(agents)} agent(s) discovered. Routing by AgentCard tags.")
    print()
    print("  Try:")
    print('    "100 USD to EUR"           → routes to CurrencyAgent')
    print('    "weather in Tokyo"          → routes to WeatherAgent')
    print('    "who wrote Hamlet?"         → routes to GeneralAgent')
    print()
    print("  Type 'quit' to exit, 'new' to reset conversation.")
    print()

    # One context_id per agent URL so each specialist keeps its own
    # conversation history, even across multiple A2A tasks.
    agent_contexts: dict[str, str] = {
        a.url: str(uuid.uuid4()) for a in agents
    }

    async with httpx.AsyncClient(timeout=60.0) as http:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not user_input or user_input.lower() == "quit":
                print("Bye!")
                break
            if user_input.lower() == "new":
                agent_contexts = {a.url: str(uuid.uuid4()) for a in agents}
                print("\n--- New conversation (all contexts reset) ---\n")
                continue

            # Route the query
            target = route_query(user_input, agents)

            if not target:
                print("  No suitable agent found for this query.\n")
                continue

            print(f"  [Routing to {target.name} at {target.url}]")

            # Send directly to the chosen agent with a persistent context_id
            ctx_id = agent_contexts.get(target.url, str(uuid.uuid4()))
            a2a_client = A2AClient(httpx_client=http, url=target.url)
            request = SendMessageRequest(
                id=str(uuid.uuid4()),
                params=MessageSendParams(
                    message=A2AMessage(
                        role=Role.user,
                        message_id=str(uuid.uuid4()),
                        parts=[TextPart(text=user_input)],
                        context_id=ctx_id,
                    ),
                ),
            )

            try:
                response = await a2a_client.send_message(request)
            except Exception as exc:
                print(f"  Error: {exc}\n")
                continue

            result = response.root

            if hasattr(result, "error") and result.error:
                print(f"  Error: {result.error}\n")
                continue

            text = _extract_response_text(result.result)
            print(f"Agent ({target.name}): {text}\n")


if __name__ == "__main__":
    asyncio.run(chat())
