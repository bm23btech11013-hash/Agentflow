"""Skill activation / deactivation tools and marker extraction.

Provides factory functions that create the ``set_skill`` and ``clear_skill``
tools the LLM can call, plus a helper to scan tool-result messages for the
activation markers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentflow.state.message import Message
    from agentflow.skills.registry import SkillsRegistry

from agentflow.skills.models import SkillConfig

logger = logging.getLogger("agentflow.skills.activation")

_ACTIVATED_PREFIX = "SKILL_ACTIVATED:"
_DEACTIVATED_MARKER = "SKILL_DEACTIVATED"


def make_set_skill_tool(
    registry: "SkillsRegistry",
    config: SkillConfig | None = None,
) -> Callable:
    """Factory that returns a ``set_skill`` function whose doc-string
    lists the available skills.

    When the LLM calls ``set_skill("triage")``, the function returns
    ``"SKILL_ACTIVATED:triage"`` — a marker string that the framework
    intercepts in ``InvokeNodeHandler._call_tools``.
    """
    from agentflow.skills.registry import SkillsRegistry  # avoid circular import

    available = registry.get_all()
    names_str = ", ".join(sorted(m.name for m in available))
    skill_list = "\n".join(f"- {m.name}: {m.description}" for m in available)

    def set_skill(skill_name: str) -> str:  # noqa: D401
        """Activate a specialized skill protocol.

        Call this tool when the user's request matches a skill domain.
        After activation, the skill's instructions and tools will be
        injected automatically.

        Args:
            skill_name: Name of the skill to activate.
        """
        if registry.get(skill_name) is not None:
            logger.info("Skill activated via tool: '%s'", skill_name)
            return f"{_ACTIVATED_PREFIX}{skill_name}"

        logger.warning("Unknown skill requested: %r. Available: %s", skill_name, names_str)
        return f"ERROR: Unknown skill '{skill_name}'. Available: {names_str}"

    # Patch the docstring so the LLM sees available skills
    set_skill.__doc__ = (
        "Activate a specialized skill protocol.\n\n"
        "Call this tool when the user's request matches a skill domain.\n"
        "Call this to activate a skill OR to switch away from an already-active skill "
        "when the user's request better matches a different one.\n\n"
        f"Available skills:\n{skill_list}\n\n"
        "Args:\n"
        "    skill_name: Name of the skill to activate."
    )
    return set_skill


def make_clear_skill_tool() -> Callable:
    """Factory that returns a ``clear_skill`` tool to deactivate the
    current skill."""

    def clear_skill() -> str:
        """Deactivate the currently active skill and return to general mode."""
        logger.info("Skill deactivated via tool")
        return _DEACTIVATED_MARKER

    return clear_skill


def extract_skill_markers(
    messages: list["Message"],
) -> tuple[list[str], bool]:
    """Scan a list of tool-result messages for skill activation/deactivation markers.

    Returns:
        ``(activated_names, should_deactivate)``
        where *activated_names* is a list of skill names found in
        ``SKILL_ACTIVATED:<name>`` markers and *should_deactivate* is ``True``
        if a ``SKILL_DEACTIVATED`` marker was found.
    """
    activated: list[str] = []
    should_deactivate = False

    for msg in messages:
        if not hasattr(msg, "content") or msg.content is None:
            continue
        for block in msg.content:
            raw = getattr(block, "output", None) or getattr(block, "text", None)
            if raw is None:
                raw = str(block)
            if not isinstance(raw, str):
                continue
            if raw.startswith(_ACTIVATED_PREFIX):
                name = raw[len(_ACTIVATED_PREFIX):].strip()
                if name:
                    activated.append(name)
            elif raw == _DEACTIVATED_MARKER:
                should_deactivate = True

    return activated, should_deactivate
