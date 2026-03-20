"""Data models for the Agentflow Skills system.

Defines SkillMeta (parsed from SKILL.md frontmatter) and SkillConfig
(user-facing configuration for enabling skills on an Agent).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SkillMeta(BaseModel):
    """Metadata about a single skill, parsed from SKILL.md frontmatter."""

    name: str
    description: str
    triggers: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    tags: set[str] = Field(default_factory=set)
    priority: int = 0
    skill_dir: str = ""
    skill_file: str = ""

    class Config:
        frozen = False


class SkillConfig(BaseModel):
    """Configuration for the skills system on an Agent."""

    skills_dir: str | None = None
    max_active: int = 1
    auto_deactivate: bool = True
    inject_trigger_table: bool = True
    hot_reload: bool = True
