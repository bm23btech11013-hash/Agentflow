"""Skills support for Agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agentflow.graph.tool_node import ToolNode


if TYPE_CHECKING:
    from agentflow.skills.injection import SkillInjector
    from agentflow.skills.models import SkillConfig
    from agentflow.skills.registry import SkillsRegistry


logger = logging.getLogger("agentflow.agent")


class AgentSkillsMixin:
    """Skills registration and injection helpers for Agent."""

    # Instance attributes set by _setup_skills
    _skills_config: SkillConfig | None
    _skills_registry: SkillsRegistry | None
    _skill_injector: SkillInjector | None
    _tool_node: ToolNode | None

    def _setup_skills(self, skills: SkillConfig | None) -> None:
        """Initialize skills infrastructure if a SkillConfig is provided.

        Args:
            skills: Optional SkillConfig instance with skills_dir and options.
        """
        self._skills_config = None
        self._skills_registry = None
        self._skill_injector = None

        if skills is None:
            return

        from agentflow.skills.activation import make_clear_skill_tool, make_set_skill_tool
        from agentflow.skills.injection import SkillInjector
        from agentflow.skills.models import SkillConfig
        from agentflow.skills.registry import SkillsRegistry

        self._skills_config = skills if isinstance(skills, SkillConfig) else SkillConfig()
        self._skills_registry = SkillsRegistry()

        if self._skills_config.skills_dir:
            self._skills_registry.discover(self._skills_config.skills_dir)

        self._skill_injector = SkillInjector(self._skills_registry, config=self._skills_config)

        # Create skill tools
        set_skill_fn = make_set_skill_tool(self._skills_registry, self._skills_config)
        clear_skill_fn = make_clear_skill_tool()

        # Add skill tools to the tool node
        if self._tool_node is None:
            self._tool_node = ToolNode([set_skill_fn, clear_skill_fn])
        else:
            self._tool_node._funcs[set_skill_fn.__name__] = set_skill_fn
            self._tool_node._funcs[clear_skill_fn.__name__] = clear_skill_fn

        logger.info(
            "Skills enabled: %d skill(s) discovered",
            len(self._skills_registry.names()),
        )

    def _build_skill_prompts(
        self,
        state: Any,
        system_prompt: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Build effective system prompts with skills injected.

        Args:
            state: Current AgentState with execution_meta containing active_skills.
            system_prompt: Base system prompt list.

        Returns:
            Tuple of (effective_system_prompt, skill_extra_messages).
        """
        effective_system_prompt = list(system_prompt)
        skill_extra_messages: list[dict[str, Any]] = []

        if not self._skills_config or not self._skill_injector or not self._skills_registry:
            return effective_system_prompt, skill_extra_messages

        active_skills: list[str] = state.execution_meta.internal_data.get("active_skills", [])

        if active_skills:
            # Inject active skill content as extra system messages
            skill_extra_messages = self._skill_injector.build_skill_prompts(active_skills)

        if self._skills_config.inject_trigger_table:
            trigger_table = self._skills_registry.build_trigger_table()
            if trigger_table:
                effective_system_prompt.append({"role": "system", "content": trigger_table})

        return effective_system_prompt, skill_extra_messages
