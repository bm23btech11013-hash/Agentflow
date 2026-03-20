"""Skill content injection for Agentflow.

The :class:`SkillInjector` converts currently-active skills into system
prompt dicts that are injected fresh on every LLM call via
``Agent.execute()``.  Because these messages are built on the fly (they
never enter ``state.context``), they survive context trimming.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from agentflow.skills.models import SkillConfig
from agentflow.skills.registry import SkillsRegistry

logger = logging.getLogger("agentflow.skills.injection")


class SkillInjector:
    """Builds system-prompt dicts from the currently active skill(s).

    Usage inside ``Agent.execute()``::

        injector = SkillInjector(registry)
        extra_prompts = injector.build_skill_prompts(["triage"])
        # extra_prompts → [{"role": "system", "content": "## ACTIVE SKILL: TRIAGE\\n..."}]
    """

    def __init__(
        self,
        registry: SkillsRegistry,
        config: SkillConfig | None = None,
    ) -> None:
        self._registry = registry
        self._config = config or SkillConfig()
        # Cache: skill name -> (mtime, content_str)
        self._cache: dict[str, tuple[float, str]] = {}

    def build_skill_prompts(self, active_skills: list[str]) -> list[dict[str, str]]:
        """Return a list of ``{"role": "system", "content": ...}`` dicts
        for each active skill.

        Each dict contains the SKILL.md body plus any auto-injected
        resource files, wrapped with a header so the LLM knows which
        skill is active.
        """
        prompts: list[dict[str, str]] = []

        for name in active_skills:
            meta = self._registry.get(name)
            if meta is None:
                logger.warning("Active skill '%s' not found in registry", name)
                continue

            content = self._load_with_cache(name)
            if not content:
                continue

            # Auto-inject resource files
            resources = self._registry.load_resources(name)
            for res_name, res_content in resources.items():
                content += f"\n\n---\n### Reference: {res_name}\n\n{res_content}"

            header = name.upper().replace("-", " ")
            routing_note = (
                "\n\n---\n"
                "**ROUTING:** On the next user turn, re-check the available skills table. "
                "If the request better matches a different skill, call `set_skill(other_skill_name)` "
                "immediately — do not answer using this skill."
            )
            full = f"## ACTIVE SKILL: {header}\n\n{content}{routing_note}"
            prompts.append({"role": "system", "content": full})
            logger.debug("Injecting skill prompt: '%s'", name)

        return prompts

    def get_skill_tools(self, active_skills: list[str]) -> list[Callable]:
        """Collect tool functions from the active skills' SkillMeta.

        (Currently a placeholder — skill-specific tools are a future
        enhancement tracked via ``SkillMeta`` extensions.)
        """
        # Skills discovered from the filesystem don't carry tool references.
        # This method is the extension point for when programmatic skills
        # with attached tools are supported.
        return []

    # -- internal -----------------------------------------------------------

    def _load_with_cache(self, name: str) -> str:
        """Load skill content, using an mtime-based cache when hot_reload
        is enabled to avoid re-reading unchanged files."""
        meta = self._registry.get(name)
        if meta is None:
            return ""

        if not self._config.hot_reload:
            # Hot-reload disabled — always read fresh
            return self._registry.load_content(name, hot_reload=False)

        # Check mtime for cache validity
        try:
            current_mtime = os.path.getmtime(meta.skill_file) if meta.skill_file else 0.0
        except OSError:
            current_mtime = 0.0

        cached = self._cache.get(name)
        if cached is not None and cached[0] == current_mtime:
            return cached[1]

        content = self._registry.load_content(name, hot_reload=True)
        self._cache[name] = (current_mtime, content)
        return content
