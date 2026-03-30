"""Data models for the Agentflow Skills system.

Defines SkillMeta (parsed from SKILL.md frontmatter) and SkillConfig
(user-facing configuration for enabling skills on an Agent).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator


# Skill names must be slug-like: lowercase alphanumeric, hyphens, underscores.
_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# Maximum lengths to prevent abuse.
_MAX_NAME_LEN = 128
_MAX_DESCRIPTION_LEN = 2000
_MAX_TRIGGER_LEN = 500
_MAX_TRIGGERS = 50
_MAX_RESOURCES = 100
_MAX_TAGS = 50
_MAX_PRIORITY = 1000


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

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        v = v.strip().lower()
        if not v:
            raise ValueError("Skill name must not be empty")
        if len(v) > _MAX_NAME_LEN:
            raise ValueError(f"Skill name exceeds {_MAX_NAME_LEN} characters")
        if not _SKILL_NAME_RE.match(v):
            raise ValueError(
                f"Invalid skill name '{v}'. "
                "Must be lowercase alphanumeric with hyphens/underscores, "
                "starting with a letter or digit."
            )
        return v

    @field_validator("description")
    @classmethod
    def _validate_description(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Skill description must not be empty")
        if len(v) > _MAX_DESCRIPTION_LEN:
            raise ValueError(f"Skill description exceeds {_MAX_DESCRIPTION_LEN} characters")
        return v

    @field_validator("triggers")
    @classmethod
    def _validate_triggers(cls, v: list[str]) -> list[str]:
        if len(v) > _MAX_TRIGGERS:
            raise ValueError(f"Too many triggers (max {_MAX_TRIGGERS})")
        cleaned: list[str] = []
        for t in v:
            t = t.strip()
            if not t:
                continue  # silently drop empty triggers
            if len(t) > _MAX_TRIGGER_LEN:
                raise ValueError(f"Trigger exceeds {_MAX_TRIGGER_LEN} characters: '{t[:50]}...'")
            cleaned.append(t)
        return cleaned

    @field_validator("resources")
    @classmethod
    def _validate_resources(cls, v: list[str]) -> list[str]:
        if len(v) > _MAX_RESOURCES:
            raise ValueError(f"Too many resources (max {_MAX_RESOURCES})")
        for r in v:
            r = r.strip()
            if not r:
                raise ValueError("Resource path must not be empty")
            # Path traversal protection
            if ".." in r or r.startswith("/") or r.startswith("\\"):
                raise ValueError(
                    f"Invalid resource path '{r}'. Paths must be relative and cannot contain '..'."
                )
        return v

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, v: set[str]) -> set[str]:
        if len(v) > _MAX_TAGS:
            raise ValueError(f"Too many tags (max {_MAX_TAGS})")
        return {t.strip().lower() for t in v if t.strip()}

    @field_validator("priority")
    @classmethod
    def _validate_priority(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"Priority must be non-negative, got {v}")
        if v > _MAX_PRIORITY:
            raise ValueError(f"Priority exceeds maximum ({_MAX_PRIORITY}), got {v}")
        return v


class SkillConfig(BaseModel):
    """Configuration for the skills system on an Agent."""

    skills_dir: str | None = None
    inject_trigger_table: bool = True
    hot_reload: bool = True

    @field_validator("skills_dir")
    @classmethod
    def _validate_skills_dir(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        if not v:
            raise ValueError("skills_dir must not be an empty string (use None to disable)")
        return v
