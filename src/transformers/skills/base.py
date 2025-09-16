# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Base classes and interfaces for the agentic skills framework.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


logger = logging.getLogger(__name__)


@dataclass
class SkillExecutionResult:
    """Result of executing an agentic skill."""

    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata or {}
        }

    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class SkillMetadata:
    """Metadata describing an agentic skill."""

    name: str
    description: str
    category: str
    input_types: list[str]
    output_types: list[str]
    parameters: dict[str, Any]
    examples: list[dict[str, Any]]
    tags: list[str] = None
    version: str = "1.0.0"

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class AgenticSkill(ABC):
    """
    Base class for all agentic skills.

    An agentic skill is a capability that can be discovered, configured,
    and executed by chatbots and other AI systems.
    """

    def __init__(self, skill_id: str, metadata: SkillMetadata):
        self.skill_id = skill_id
        self.metadata = metadata
        self._enabled = True

    @abstractmethod
    def execute(self, input_data: Any, **kwargs) -> SkillExecutionResult:
        """
        Execute the skill with the given input.

        Args:
            input_data: The input data for the skill
            **kwargs: Additional parameters for execution

        Returns:
            SkillExecutionResult containing the execution result
        """
        pass

    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data for the skill.

        Args:
            input_data: The input data to validate

        Returns:
            True if input is valid, False otherwise
        """
        return True

    def get_schema(self) -> dict[str, Any]:
        """
        Get the JSON schema for this skill.

        Returns:
            Dictionary containing the skill schema
        """
        return {
            "skill_id": self.skill_id,
            "name": self.metadata.name,
            "description": self.metadata.description,
            "category": self.metadata.category,
            "input_types": self.metadata.input_types,
            "output_types": self.metadata.output_types,
            "parameters": self.metadata.parameters,
            "examples": self.metadata.examples,
            "tags": self.metadata.tags,
            "version": self.metadata.version,
            "enabled": self._enabled
        }

    def enable(self):
        """Enable this skill."""
        self._enabled = True

    def disable(self):
        """Disable this skill."""
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Check if skill is enabled."""
        return self._enabled


class SkillRegistry:
    """
    Registry for managing agentic skills.

    Provides discovery, registration, and retrieval of skills.
    """

    def __init__(self):
        self._skills: dict[str, AgenticSkill] = {}
        self._categories: dict[str, list[str]] = {}

    def register(self, skill: AgenticSkill) -> None:
        """
        Register a skill in the registry.

        Args:
            skill: The skill to register
        """
        if skill.skill_id in self._skills:
            logger.warning(f"Skill '{skill.skill_id}' is already registered. Overwriting.")

        self._skills[skill.skill_id] = skill

        # Update category index
        category = skill.metadata.category
        if category not in self._categories:
            self._categories[category] = []

        if skill.skill_id not in self._categories[category]:
            self._categories[category].append(skill.skill_id)

        logger.info(f"Registered skill '{skill.skill_id}' in category '{category}'")

    def unregister(self, skill_id: str) -> bool:
        """
        Unregister a skill from the registry.

        Args:
            skill_id: ID of the skill to unregister

        Returns:
            True if skill was unregistered, False if not found
        """
        if skill_id not in self._skills:
            return False

        skill = self._skills[skill_id]
        category = skill.metadata.category

        # Remove from skills
        del self._skills[skill_id]

        # Remove from category index
        if category in self._categories:
            if skill_id in self._categories[category]:
                self._categories[category].remove(skill_id)

            # Clean up empty categories
            if not self._categories[category]:
                del self._categories[category]

        logger.info(f"Unregistered skill '{skill_id}'")
        return True

    def get_skill(self, skill_id: str) -> Optional[AgenticSkill]:
        """
        Get a skill by ID.

        Args:
            skill_id: ID of the skill to retrieve

        Returns:
            The skill if found, None otherwise
        """
        return self._skills.get(skill_id)

    def list_skills(self, category: Optional[str] = None, enabled_only: bool = True) -> list[AgenticSkill]:
        """
        List available skills.

        Args:
            category: Filter by category (optional)
            enabled_only: Only return enabled skills

        Returns:
            List of skills matching the criteria
        """
        skills = list(self._skills.values())

        if category:
            skills = [skill for skill in skills if skill.metadata.category == category]

        if enabled_only:
            skills = [skill for skill in skills if skill.enabled]

        return skills

    def list_categories(self) -> list[str]:
        """
        List available skill categories.

        Returns:
            List of category names
        """
        return list(self._categories.keys())

    def search_skills(self, query: str) -> list[AgenticSkill]:
        """
        Search for skills by name, description, or tags.

        Args:
            query: Search query

        Returns:
            List of matching skills
        """
        query_lower = query.lower()
        results = []

        for skill in self._skills.values():
            if not skill.enabled:
                continue

            # Search in name and description
            if (query_lower in skill.metadata.name.lower() or
                query_lower in skill.metadata.description.lower()):
                results.append(skill)
                continue

            # Search in tags
            for tag in skill.metadata.tags:
                if query_lower in tag.lower():
                    results.append(skill)
                    break

        return results

    def get_schema(self) -> dict[str, Any]:
        """
        Get the complete registry schema.

        Returns:
            Dictionary containing all skills and their schemas
        """
        return {
            "skills": {skill_id: skill.get_schema() for skill_id, skill in self._skills.items()},
            "categories": self._categories,
            "total_skills": len(self._skills),
            "enabled_skills": len([s for s in self._skills.values() if s.enabled])
        }
