"""Central skills registry for Agentflow.

The registry is the single index of all discovered skills.  It can be
registered in InjectQ so that any graph node can access it.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from agentflow.skills.loader import (
    discover_skills,
    load_resource,
    load_skill_content,
)
from agentflow.skills.models import SkillMeta


logger = logging.getLogger("agentflow.skills.registry")


class SkillsRegistry:
    """Central registry that holds discovered :class:`SkillMeta` entries.

    Typical lifecycle::

        registry = SkillsRegistry()
        registry.discover("/path/to/skills")
        table = registry.build_trigger_table()
        tool = registry.build_set_skill_tool()
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillMeta] = {}
        # file mod-time cache for hot-reload optimisation
        self._mtimes: dict[str, float] = {}

    # -- registration -------------------------------------------------------

    def register(self, meta: SkillMeta) -> None:
        """Register a single :class:`SkillMeta`.

        Duplicate names are allowed only when re-registering the exact same
        skill file (idempotent registration). Different files using the same
        skill name raise a ValueError to avoid silent overrides.
        """
        existing = self._skills.get(meta.name)
        if existing is not None:
            if existing.skill_file == meta.skill_file:
                logger.debug("Skill '%s' already registered from %s", meta.name, meta.skill_file)
                return
            raise ValueError(
                f"Duplicate skill name '{meta.name}' from {meta.skill_file} "
                f"(already registered from {existing.skill_file})"
            )

        self._skills[meta.name] = meta
        if meta.skill_file:
            with contextlib.suppress(OSError):
                self._mtimes[meta.name] = Path(meta.skill_file).stat().st_mtime
        logger.info("Registered skill: '%s'", meta.name)

    def discover(self, skills_dir: str) -> list[SkillMeta]:
        """Auto-discover skills from *skills_dir* and register them."""
        found = discover_skills(skills_dir)
        for meta in found:
            self.register(meta)
        return found

    # -- lookup -------------------------------------------------------------

    def get(self, name: str) -> SkillMeta | None:
        return self._skills.get(name)

    def get_all(self, tags: set[str] | None = None) -> list[SkillMeta]:
        skills = list(self._skills.values())
        if tags:
            skills = [s for s in skills if s.tags & tags]
        return skills

    def names(self) -> list[str]:
        return sorted(self._skills.keys())

    # -- content loading ----------------------------------------------------

    def load_content(self, name: str, hot_reload: bool = True) -> str:
        """Load the body of a skill's SKILL.md.

        When *hot_reload* is ``True``, file mtime is refreshed before loading
        content. This supports edit-aware behavior without changing the return
        contract: this method always returns the current skill content string.
        """
        meta = self._skills.get(name)
        if meta is None:
            return ""

        if hot_reload and meta.skill_file:
            try:
                current_mtime = Path(meta.skill_file).stat().st_mtime
            except OSError:
                current_mtime = 0.0
            cached = self._mtimes.get(name, 0.0)
            # Always load on first access (cached == current when just registered)
            if cached != 0.0 and current_mtime == cached:
                # File unchanged -- but we still return content (caller needs it)
                pass
            else:
                self._mtimes[name] = current_mtime

        return load_skill_content(meta)

    def load_resources(self, name: str) -> dict[str, str]:
        """Load all resource files for a skill.  Returns ``{filename: content}``."""
        meta = self._skills.get(name)
        if meta is None:
            return {}

        result: dict[str, str] = {}
        for rel_path in meta.resources:
            content = load_resource(meta, rel_path)
            if content is not None:
                result[Path(rel_path).name] = content
        return result

    # -- prompt helpers -----------------------------------------------------

    def build_trigger_table(self, tags: set[str] | None = None) -> str:
        """Generate a markdown trigger table for the LLM system prompt."""
        skills = sorted(
            self.get_all(tags=tags),
            key=lambda s: (-s.priority, s.name),
        )
        if not skills:
            return ""

        lines = [
            "## Available Skills\n",
            "On EVERY user turn, check whether the request matches a skill below.\n"
            "If it matches a skill, call `set_skill(skill_name)` before finalizing your answer.\n",
            "| Skill | When to use |",
            "|-------|-------------|",
        ]
        for meta in skills:
            desc = meta.description
            if meta.triggers:
                desc = "; ".join(meta.triggers[:3])
            safe_desc = desc.replace("|", "\\|").replace("\n", " ").strip()
            lines.append(f"| `{meta.name}` | {safe_desc} |")

        return "\n".join(lines)

    def build_set_skill_tool(self, hot_reload: bool = True) -> Any:
        """Convenience — delegates to :func:`activation.make_set_skill_tool`."""
        from agentflow.skills.activation import make_set_skill_tool

        return make_set_skill_tool(self, hot_reload=hot_reload)
