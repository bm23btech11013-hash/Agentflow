"""
A2A server for the currency conversion agent.

Exposes the agentflow currency graph as a standard A2A HTTP endpoint.

Usage::

    python -m examples.a2a_sdk.currency.server
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from a2a.types import AgentSkill

from agentflow.a2a_integration import create_a2a_server, make_agent_card
from examples.a2a_sdk.currency.agent import build_currency_graph

if __name__ == "__main__":
    compiled = build_currency_graph()

    card = make_agent_card(
        name="CurrencyAgent",
        description=(
            "A currency conversion agent that provides real-time exchange "
            "rates using the Frankfurter API. Ask: 'How much is 10 USD in INR?'"
        ),
        url="http://localhost:10000/",
        skills=[
            AgentSkill(
                id="currency_conversion",
                name="Currency Conversion",
                description="Convert between currencies using live exchange rates",
                tags=["currency", "finance", "exchange-rate"],
                examples=[
                    "How much is 100 USD in EUR?",
                    "Convert 50 GBP to JPY",
                    "What is the exchange rate from AUD to CAD?",
                ],
            )
        ],
    )

    print("Starting CurrencyAgent A2A server on http://localhost:10000")
    print("Agent card: http://localhost:10000/.well-known/agent-card.json")
    print("Press Ctrl+C to stop.\n")

    create_a2a_server(compiled, card, port=10000)
