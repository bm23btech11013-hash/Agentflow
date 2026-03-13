"""
A2A SDK Client Example — call the running A2A server.

Make sure the server is running first::

    python -m examples.a2a_sdk.server

Then run this client::

    python -m examples.a2a_sdk.client
"""

import asyncio

from agentflow.a2a_integration import delegate_to_a2a_agent

AGENT_URL = "http://localhost:9999"


async def main():
    print(f"Sending message to A2A agent at {AGENT_URL} ...")

    response = await delegate_to_a2a_agent(
        AGENT_URL,
        "Hello from the A2A SDK client!",
    )

    print(f"Agent response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
