"""
A2A client for the currency conversion agent.

Sends sample currency queries to the running A2A server and prints
the responses.

Usage::

    python -m examples.a2a_sdk.currency.client
"""

from __future__ import annotations

import asyncio

from agentflow.a2a_integration import delegate_to_a2a_agent

AGENT_URL = "http://localhost:10000"

QUERIES = [
    "How much is 10 USD in INR?",
    "What about 50 EUR to GBP?",
    "Convert 1000 JPY to USD",
]


async def main():
    for query in QUERIES:
        print(f"\nQuery: {query}")
        response = await delegate_to_a2a_agent(AGENT_URL, query)
        print(f"Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
