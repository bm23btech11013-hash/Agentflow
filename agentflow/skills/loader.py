"""Filesystem-based skill loader for Agentflow.

Scans a directory for SKILL.md files with YAML frontmatter, parses them
into SkillMeta objects, and provides content loading utilities.

Directory structure:
    skills/
    +-- triage/
    |   +-- SKILL.md          # YAML frontmatter + markdown body
    |   +-- protocols.md      # resource file referenced in frontmatter
    +-- prescription/
        +-- SKILL.md
        +-- formulary.md
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from agentflow.skills.models import SkillMeta


logger = logging.getLogger("agentflow.skills.loader")


def discover_skills(skills_dir: str) -> list[SkillMeta]:  # noqa: PLR0912
    """Scan *skills_dir* for subdirectories containing a ``SKILL.md`` with
    valid YAML frontmatter and return a list of :class:`SkillMeta`.

    Each subdirectory that contains a ``SKILL.md`` with at least ``name``
    and ``description`` in its frontmatter is registered.  ``triggers``,
    ``resources``, ``tags``, and ``priority`` are optional and can be placed
    either at the top level **or** inside a ``metadata`` block (preferred —
    avoids IDE schema warnings)::

        ---
        name: skill-name
        description: When to use this skill (1-3 sentences)
        metadata:
          triggers:
            - user says X
            - user says Y
          resources:
            - protocols.md
          tags:
            - domain
          priority: 5
        ---
    """
    results: list[SkillMeta] = []

    skills_path = Path(skills_dir)
    if not skills_path.is_dir():
        logger.warning("Skills directory not found: %s", skills_dir)
        return results

    for entry in sorted(p.name for p in skills_path.iterdir()):
        skill_dir = skills_path / entry
        skill_file = skill_dir / "SKILL.md"

        if not skill_dir.is_dir() or not skill_file.is_file():
            continue

        frontmatter = _parse_frontmatter(str(skill_file))
        if frontmatter is None:
            logger.warning("Skipping '%s': no valid YAML frontmatter in SKILL.md", entry)
            continue

        name = str(frontmatter.get("name", "")).strip()
        description = str(frontmatter.get("description", "")).strip()
        if not name or not description:
            logger.warning("Skipping '%s': frontmatter missing 'name' or 'description'", entry)
            continue

        # Optional fields can live inside `metadata:` (preferred — avoids IDE
        # schema warnings) or at the top level (backwards compatible).
        meta_block: dict[str, Any] = frontmatter.get("metadata") or {}

        triggers = meta_block.get("triggers") or frontmatter.get("triggers", [])
        if isinstance(triggers, str):
            triggers = [triggers]
        elif isinstance(triggers, list):
            triggers = [str(t).strip() for t in triggers if str(t).strip()]
        else:
            triggers = []

        resources: list[str] = []
        for rel_path in meta_block.get("resources") or frontmatter.get("resources", []):
            abs_path = skill_dir / rel_path
            if abs_path.is_file():
                resources.append(rel_path)
            else:
                logger.warning("Resource not found for skill '%s': %s", name, rel_path)

        raw_tags = meta_block.get("tags") or frontmatter.get("tags", [])
        tags = (
            {str(t).strip() for t in raw_tags if str(t).strip()}
            if isinstance(raw_tags, list)
            else set()
        )

        raw_priority = meta_block.get("priority")
        if raw_priority is None:
            raw_priority = frontmatter.get("priority", 0)
        try:
            priority = int(raw_priority)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid priority for skill '%s': %r (defaulting to 0)",
                name,
                raw_priority,
            )
            priority = 0

        results.append(
            SkillMeta(
                name=name,
                description=description,
                triggers=triggers,
                resources=resources,
                tags=tags,
                priority=priority,
                skill_dir=str(skill_dir),
                skill_file=str(skill_file),
            )
        )
        logger.info("Discovered skill: '%s' (%d resource(s))", name, len(resources))

    return results


def load_skill_content(meta: SkillMeta) -> str:
    """Read and return the body of a SKILL.md (stripping YAML frontmatter)."""
    try:
        content = Path(meta.skill_file).read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Cannot read %s: %s", meta.skill_file, exc)
        return ""

    # Strip frontmatter
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            content = content[end + 4 :].lstrip("\n")

    return content


def load_resource(meta: SkillMeta, rel_path: str) -> str | None:
    """Read a resource file relative to the skill directory."""
    abs_path = Path(meta.skill_dir) / rel_path
    try:
        return abs_path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Could not read resource: %s", abs_path)
        return None


# -- internal ---------------------------------------------------------------


def _parse_frontmatter(skill_file: str) -> dict[str, Any] | None:
    """Extract and parse YAML frontmatter from a SKILL.md file."""
    try:
        content = Path(skill_file).read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("Cannot read %s: %s", skill_file, exc)
        return None

    if not content.startswith("---"):
        return None

    end = content.find("\n---", 3)
    if end == -1:
        return None

    raw_yaml = content[3:end].strip()
    try:
        return yaml.safe_load(raw_yaml) or {}
    except yaml.YAMLError as exc:
        logger.error("YAML parse error in %s: %s", skill_file, exc)
        return None
