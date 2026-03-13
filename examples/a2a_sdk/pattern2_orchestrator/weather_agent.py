"""
Pattern 2 — WeatherAgent specialist (port 10001).

A specialist agent that receives COMPLETE weather queries from the
OrchestratorAgent. It never interacts with the human directly.

Uses the Open-Meteo API (free, no API key needed) for weather data
and a simple geocoding endpoint for city→coordinates.

Usage::

    python -m examples.a2a_sdk.pattern2_orchestrator.weather_agent
"""

from __future__ import annotations

import logging

import httpx
import uvicorn
from dotenv import load_dotenv
from litellm import acompletion

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill

from agentflow.a2a_integration.executor import AgentFlowExecutor
from agentflow.a2a_integration.server import make_agent_card
from agentflow.adapters.llm.model_response_converter import ModelResponseConverter
from agentflow.graph import StateGraph, ToolNode
from agentflow.graph.compiled_graph import CompiledGraph
from agentflow.state import AgentState
from agentflow.utils.constants import END
from agentflow.utils.converter import convert_messages

load_dotenv()
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Tool — Open-Meteo API (free, no key needed)                         #
# ------------------------------------------------------------------ #


async def get_weather(
    city: str,
) -> dict:
    """Get current weather for a city using Open-Meteo API.

    Args:
        city: City name (e.g. 'London', 'New York', 'Tokyo').

    Returns:
        dict with weather information including temperature,
        wind speed, and weather description.
    """
    # Step 1: Geocode city name to lat/lon
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    async with httpx.AsyncClient() as client:
        geo_resp = await client.get(geo_url, params={"name": city, "count": 1})
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        if not geo_data.get("results"):
            return {"error": f"Could not find city: {city}"}

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        name = geo_data["results"][0]["name"]
        country = geo_data["results"][0].get("country", "")

        # Step 2: Get current weather
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_resp = await client.get(
            weather_url,
            params={
                "latitude": lat,
                "longitude": lon,
                "current_weather": True,
            },
        )
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()

        current = weather_data.get("current_weather", {})

        # Map WMO weather codes to descriptions
        wmo_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy",
            3: "Overcast", 45: "Foggy", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Slight rain showers", 81: "Moderate rain showers",
            82: "Violent rain showers", 95: "Thunderstorm",
        }
        weather_code = current.get("weathercode", 0)
        description = wmo_codes.get(weather_code, f"Code {weather_code}")

        return {
            "city": name,
            "country": country,
            "temperature_celsius": current.get("temperature"),
            "wind_speed_kmh": current.get("windspeed"),
            "weather": description,
            "time": current.get("time"),
        }


tool_node = ToolNode([get_weather])

# ------------------------------------------------------------------ #
#  LLM node                                                            #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """\
You are a weather specialist.
You receive COMPLETE weather queries (city name provided).
Use the get_weather tool to look up current weather and return a
clear, friendly weather report.
Do NOT ask follow-up questions — you always receive complete info.\
"""


async def llm_node(state: AgentState):
    messages = convert_messages(
        system_prompts=[{"role": "system", "content": SYSTEM_PROMPT}],
        state=state,
    )

    if state.context and state.context[-1].role == "tool":
        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
        )
    else:
        tools = await tool_node.all_tools()
        response = await acompletion(
            model="gemini/gemini-2.5-flash",
            messages=messages,
            tools=tools,
        )

    return ModelResponseConverter(response, converter="litellm")


# ------------------------------------------------------------------ #
#  Routing                                                             #
# ------------------------------------------------------------------ #


def should_use_tools(state: AgentState) -> str:
    if not state.context or len(state.context) == 0:
        return "TOOL"

    last_message = state.context[-1]

    if (
        hasattr(last_message, "tools_calls")
        and last_message.tools_calls
        and len(last_message.tools_calls) > 0
        and last_message.role == "assistant"
    ):
        return "TOOL"

    if last_message.role == "tool":
        return "MAIN"

    return END


# ------------------------------------------------------------------ #
#  Build the graph                                                     #
# ------------------------------------------------------------------ #


def build_weather_graph() -> CompiledGraph:
    graph = StateGraph()
    graph.add_node("MAIN", llm_node)
    graph.add_node("TOOL", tool_node)

    graph.add_conditional_edges(
        "MAIN",
        should_use_tools,
        {"TOOL": "TOOL", "MAIN": "MAIN", END: END},
    )
    graph.add_edge("TOOL", "MAIN")
    graph.set_entry_point("MAIN")

    return graph.compile()


# ------------------------------------------------------------------ #
#  Server                                                              #
# ------------------------------------------------------------------ #


def run_server() -> None:
    """Start WeatherAgent on port 10001."""
    compiled = build_weather_graph()

    card = make_agent_card(
        name="WeatherAgent",
        description=(
            "Specialist weather agent. Receives complete queries and "
            "returns current weather reports using the Open-Meteo API."
        ),
        url="http://localhost:10001/",
        skills=[
            AgentSkill(
                id="weather_lookup",
                name="Weather Lookup",
                description="Get current weather for any city in the world.",
                tags=["weather", "forecast", "temperature"],
                examples=[
                    "What's the weather in London?",
                    "Weather in Tokyo tomorrow",
                    "Is it raining in New York?",
                ],
            )
        ],
    )

    handler = DefaultRequestHandler(
        agent_executor=AgentFlowExecutor(compiled),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=card, http_handler=handler)

    print("  [WeatherAgent] Running on http://localhost:10001")
    print("  Agent card: http://localhost:10001/.well-known/agent-card.json")
    print("  Press Ctrl+C to stop.\n")

    uvicorn.run(app.build(), host="0.0.0.0", port=10001)


if __name__ == "__main__":
    run_server()
