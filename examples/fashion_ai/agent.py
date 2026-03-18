"""Fashion sales agent wired to catalog MCP tools.

This graph powers a single sales-focused agent for an Indian fashion storefront.
It relies on the catalog MCP server for retrieval, so the agent can stay
grounded in live product data while behaving like an assertive sari salesperson.
"""

from __future__ import annotations

import logging
import os
from datetime import date
from decimal import Decimal
from textwrap import dedent
from typing import Annotated

from dotenv import load_dotenv
from injectq import Inject, InjectQ
from pydantic import Field

from agentflow.checkpointer import InMemoryCheckpointer
from agentflow.graph import Agent, CompiledGraph, StateGraph, ToolNode
from agentflow.state import AgentState, Message
from agentflow.utils import tool
from agentflow.utils.constants import END


container = InjectQ.get_instance()


class CatalogMCPService:
    async def list_fabrics(self) -> list[str]:
        """Return unique fabrics across active products."""
        return ["silk", "cotton", "georgette", "chiffon"]

    async def list_colors(self) -> list[str]:
        """Return unique colors across active products."""
        return ["red", "blue", "green", "yellow", "pink", "orange"]

    async def list_categories(self) -> list[dict]:
        """Return active categories and subcategories."""
        return [
            {
                "name": "sarees",
                "subcategories": ["silk sarees", "cotton sarees", "georgette sarees"],
            },
            {
                "name": "lehengas",
                "subcategories": ["bridal lehengas", "party lehengas"],
            },
            {
                "name": "salwar kameez",
                "subcategories": ["anarkali suits", "churidar suits"],
            },
        ]

    async def search_products(  # noqa: PLR0913
        self,
        text: str = "",
        category: str | None = None,
        subcategory: str | None = None,
        fabric: str | None = None,
        color: str | None = None,
        min_price: Decimal | None = None,
        max_price: Decimal | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict:
        """Search products by text, price range, category, subcategory, fabric, and color."""
        # In a real implementation, this would query the catalog database with filters.
        # Here we return dummy data for demonstration.
        dummy_product = {
            "id": "123",
            "name": "Elegant Silk Saree",
            "category": "sarees",
            "subcategory": "silk sarees",
            "fabric": "silk",
            "color": "red",
            "price": Decimal("199.99"),
        }
        return {
            "total": 1,
            "items": [dummy_product],
        }


# register the service in the container
container.bind_instance(CatalogMCPService, CatalogMCPService())


@tool(
    name="list_catalog_categories",
    description=(
        "Return all active product categories and their subcategories "
        "for catalog browsing and filtering."
    ),
    tags={"catalog", "categories", "read"},
    metadata={"domain": "catalog", "entity": "category", "operation": "list"},
)
async def categories(
    config: dict,
    service: CatalogMCPService = Inject[CatalogMCPService],
) -> list[dict]:
    """Return all active categories with their subcategories."""
    return await service.list_categories()


@tool(
    name="list_product_fabrics",
    description=(
        "Return the unique fabric values available across active products "
        "for filtering and discovery."
    ),
    tags={"catalog", "fabrics", "filters", "read"},
    metadata={
        "domain": "catalog",
        "entity": "product",
        "facet": "fabric",
        "operation": "list",
    },
)
async def fabrics(
    config: dict,
    service: CatalogMCPService = Inject[CatalogMCPService],
) -> dict:
    """Return unique product fabrics."""
    items = await service.list_fabrics()
    return {"total": len(items), "items": items}


@tool(
    name="list_product_colors",
    description=(
        "Return the unique color values available across active products "
        "for filtering and discovery."
    ),
    tags={"catalog", "colors", "filters", "read"},
)
async def colors(
    config: dict,
    service: CatalogMCPService = Inject[CatalogMCPService],
) -> dict:
    """Return unique product colors."""
    items = await service.list_colors()
    return {"total": len(items), "items": items}


@tool(
    name="search_catalog_products",
    description=(
        "Search active products by text, category, subcategory, fabric, color, and price range."
    ),
    tags={"catalog", "products", "search", "read"},
    metadata={"domain": "catalog", "entity": "product", "operation": "search"},
)
async def search_products(  # noqa: PLR0913
    config: dict,
    text: str = "",
    category: str | None = None,
    subcategory: str | None = None,
    fabric: str | None = None,
    color: str | None = None,
    min_price: Decimal | None = None,
    max_price: Decimal | None = None,
    limit: Annotated[int, Field(ge=1, le=100)] = 20,
    offset: Annotated[int, Field(ge=0)] = 0,
    service: CatalogMCPService = Inject[CatalogMCPService],
) -> dict:
    """Search products by text, price range, category, subcategory, fabric, and color."""
    return await service.search_products(
        text=text,
        category=category,
        subcategory=subcategory,
        fabric=fabric,
        color=color,
        min_price=min_price,
        max_price=max_price,
        limit=limit,
        offset=offset,
    )


# Silence logging from this module by default.
# AgentFlow and the underlying libraries may emit info logs; we keep this module quiet.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.propagate = False

load_dotenv()

checkpointer = InMemoryCheckpointer()

FASHION_AGENT_MODEL = os.getenv("FASHION_AGENT_MODEL", "gemini-2.5-flash")
FASHION_AGENT_PROVIDER = os.getenv("FASHION_AGENT_PROVIDER", "google")


SALES_PROMPT = dedent(
    f"""
    You are an assertive Indian fashion sales consultant for a virtual sari shop named 
    Triguna Silks.
    Today is {date.today().isoformat()}.

    Your job is to sell confidently, move the shopper forward quickly, and keep the
    conversation focused on showing relevant products from the live catalog.

    Behavior rules:
    - Be proactive. Show options early instead of waiting for long back-and-forth.
    - Default to sari recommendations first unless the shopper clearly asks for a different item.
    - Use the catalog tools for product retrieval whenever the shopper asks for products,
      preferences, budget, colors, fabric, or alternatives.
    - If the shopper rejects a design, immediately pivot and show different options with a
      new color, fabric, style direction, or price point.
    - Ask at most one short clarification question only when it materially improves the
      recommendation, such as occasion, budget, preferred color, or fabric.
    - If details are missing, make a reasonable first pass, show options, and then refine.
    - Keep the tone sales-forward, polished, and slightly pushy, but never rude.
    - Never invent catalog items, prices, colors, or fabrics. Only use tool results.
    - If there are no exact matches, broaden the search and explain the closest alternatives.

    Recommendation style:
    - When showing products, present 3 to 5 options when possible.
    - For each option, mention the product name or identifier, color, fabric, price,
      and a short reason it suits the shopper.
    - Compare options briefly so the shopper can decide fast.
    - End recommendation turns with a direct next step.

    Virtual try-on policy:
    - After you present products, always invite the shopper to use virtual try-on.
    - Say clearly that they can select the virtual try-on option for the design they like.
    - If the shopper likes one item more than the rest, explicitly push that item toward
      virtual try-on.

    Tool usage guide:
    - Use list_catalog_categories when you need category or subcategory awareness.
    - Use list_product_fabrics and list_product_colors to discover valid filter values.
    - Use search_catalog_products as the main retrieval tool for recommendations.
    - Prefer grounded retrieval over general fashion advice.

    Response shape:
    - Keep answers concise, commercial, and product-first.
    - Do not dump raw tool output.
    - Convert tool results into a persuasive showroom-style response.
    """
).strip()


tool_node = ToolNode(
    tools=[
        categories,
        fabrics,
        colors,
        search_products,
    ],
)

agent = Agent(
    model=FASHION_AGENT_MODEL,
    provider=FASHION_AGENT_PROVIDER,
    system_prompt=[
        {
            "role": "system",
            "content": SALES_PROMPT,
        }
    ],
    tool_node_name="TOOL",
    trim_context=True,
    temperature=0.7,
    reasoning_config=None,
)


def should_use_tools(state: AgentState) -> str:
    """Route between the assistant node and the tool node."""
    if not state.context:
        return END

    last_message = state.context[-1]
    if not last_message:
        return END

    if (
        last_message.role == "assistant"
        and hasattr(last_message, "tools_calls")
        and last_message.tools_calls
    ):
        logger.debug("Routing assistant tool call to TOOL node")
        return "TOOL"

    if last_message.role == "tool":
        logger.debug("Tool result received, routing back to MAIN for final response")
        return "MAIN"

    return END


graph = StateGraph(
    container=container,
)
graph.add_node("MAIN", agent)
graph.add_node("TOOL", tool_node)
graph.add_conditional_edges(
    "MAIN",
    should_use_tools,
    {"TOOL": "TOOL", "MAIN": "MAIN", END: END},
)
graph.add_edge("TOOL", "MAIN")
graph.set_entry_point("MAIN")

app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    config = {
        "thread_id": "1",
        "recursion_limit": 25,
    }

    while True:
        user_input = input("User: ")
        is_exit = user_input.strip().lower() in {"exit", "quit", "q"}
        if is_exit:
            print("\nExiting. Thanks for shopping at Triguna Silks! \nHave a great day\n")
            break

        inp = {"messages": [Message.text_message(user_input)]}

        result = app.invoke(inp, config=config)
        for message in result["messages"]:
            print(message)
            print("\n\n")
