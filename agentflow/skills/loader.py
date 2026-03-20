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
import os
from typing import Any

import yaml

from agentflow.skills.models import SkillMeta

logger = logging.getLogger("agentflow.skills.loader")


def discover_skills(skills_dir: str) -> list[SkillMeta]:
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

    if not os.path.isdir(skills_dir):
        logger.warning("Skills directory not found: %s", skills_dir)
        return results

    for entry in sorted(os.listdir(skills_dir)):
        skill_dir = os.path.join(skills_dir, entry)
        skill_file = os.path.join(skill_dir, "SKILL.md")

        if not os.path.isdir(skill_dir) or not os.path.isfile(skill_file):
            continue

        frontmatter = _parse_frontmatter(skill_file)
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

        resources: list[str] = []
        for rel_path in (meta_block.get("resources") or frontmatter.get("resources", [])):
            abs_path = os.path.join(skill_dir, rel_path)
            if os.path.isfile(abs_path):
                resources.append(rel_path)
            else:
                logger.warning("Resource not found for skill '%s': %s", name, rel_path)

        raw_tags = meta_block.get("tags") or frontmatter.get("tags", [])
        tags = set(raw_tags) if isinstance(raw_tags, list) else set()

        priority = int(meta_block.get("priority") or frontmatter.get("priority", 0))

        results.append(
            SkillMeta(
                name=name,
                description=description,
                triggers=triggers,
                resources=resources,
                tags=tags,
                priority=priority,
                skill_dir=skill_dir,
                skill_file=skill_file,
            )
        )
        logger.info("Discovered skill: '%s' (%d resource(s))", name, len(resources))

    return results


def load_skill_content(meta: SkillMeta) -> str:
    """Read and return the body of a SKILL.md (stripping YAML frontmatter)."""
    try:
        with open(meta.skill_file, encoding="utf-8") as f:
            content = f.read()
    except OSError as exc:
        logger.warning("Cannot read %s: %s", meta.skill_file, exc)
        return ""

    # Strip frontmatter
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            content = content[end + 4:].lstrip("\n")

    return content


def load_resource(meta: SkillMeta, rel_path: str) -> str | None:
    """Read a resource file relative to the skill directory."""
    abs_path = os.path.join(meta.skill_dir, rel_path)
    try:
        with open(abs_path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        logger.warning("Could not read resource: %s", abs_path)
        return None


# -- internal ---------------------------------------------------------------

def _parse_frontmatter(skill_file: str) -> dict[str, Any] | None:
    """Extract and parse YAML frontmatter from a SKILL.md file."""
    try:
        with open(skill_file, encoding="utf-8") as f:
            content = f.read()
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
