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
Skill Manager for agentic skills.

This module provides high-level management capabilities for agentic skills,
including auto-discovery, provisioning, and integration with chatbot systems.
"""

import json
import logging
from typing import Any, Optional

from .base import AgenticSkill, SkillExecutionResult, SkillRegistry
from .pipeline_skills import create_pipeline_skills, get_available_pipeline_tasks


logger = logging.getLogger(__name__)


class SkillManager:
    """
    High-level manager for agentic skills.

    Provides auto-discovery, provisioning, and execution of skills
    for chatbot and agent systems.
    """

    def __init__(self, auto_discover: bool = True):
        """
        Initialize the skill manager.

        Args:
            auto_discover: Whether to automatically discover and register available skills
        """
        self.registry = SkillRegistry()
        self._initialized = False

        if auto_discover:
            self.discover_skills()

    def discover_skills(self) -> None:
        """
        Auto-discover and register available skills.

        This will discover pipeline-based skills and any other
        available skill implementations.
        """
        logger.info("Discovering available agentic skills...")

        try:
            # Discover pipeline skills
            self._discover_pipeline_skills()

            # Future: Add discovery for other skill types
            # self._discover_custom_skills()

            self._initialized = True
            logger.info(f"Skill discovery complete. {len(self.registry._skills)} skills registered.")

        except Exception as e:
            logger.error(f"Error during skill discovery: {e}")
            raise

    def _discover_pipeline_skills(self) -> None:
        """Discover and register pipeline-based skills."""
        try:
            # Get common pipeline tasks that are likely to work
            common_tasks = [
                "sentiment-analysis",
                "text-classification",
                "question-answering",
                "fill-mask",
                "token-classification",
                "feature-extraction"
            ]

            # Try to create skills for available tasks
            available_tasks = get_available_pipeline_tasks()
            tasks_to_create = [task for task in common_tasks if task in available_tasks]

            if not tasks_to_create:
                # If no specific tasks available, try the common ones anyway
                tasks_to_create = common_tasks

            skills = create_pipeline_skills(tasks=tasks_to_create)

            for skill in skills:
                self.registry.register(skill)

            logger.info(f"Registered {len(skills)} pipeline skills")

        except Exception as e:
            logger.warning(f"Failed to discover pipeline skills: {e}")

    def provision_skills_for_chatbot(self, categories: Optional[list[str]] = None) -> dict[str, Any]:
        """
        Provision skills for a chatbot system.

        Args:
            categories: Optional list of skill categories to include

        Returns:
            Dictionary containing provisioned skills information
        """
        if not self._initialized:
            self.discover_skills()

        # Get skills matching criteria
        if categories:
            skills = []
            for category in categories:
                skills.extend(self.registry.list_skills(category=category))
        else:
            skills = self.registry.list_skills()

        # Create provisioning information
        provisioned_skills = {
            "skills": [],
            "categories": list({skill.metadata.category for skill in skills}),
            "total_skills": len(skills),
            "execution_endpoint": "execute_skill",  # Endpoint for skill execution
            "discovery_endpoint": "discover_skills"  # Endpoint for skill discovery
        }

        for skill in skills:
            skill_info = {
                "id": skill.skill_id,
                "name": skill.metadata.name,
                "description": skill.metadata.description,
                "category": skill.metadata.category,
                "input_types": skill.metadata.input_types,
                "output_types": skill.metadata.output_types,
                "parameters": skill.metadata.parameters,
                "examples": skill.metadata.examples,
                "tags": skill.metadata.tags,
                "enabled": skill.enabled
            }
            provisioned_skills["skills"].append(skill_info)

        logger.info(f"Provisioned {len(skills)} skills for chatbot")
        return provisioned_skills

    def execute_skill(self, skill_id: str, input_data: Any, **kwargs) -> SkillExecutionResult:
        """
        Execute a skill by ID.

        Args:
            skill_id: ID of the skill to execute
            input_data: Input data for the skill
            **kwargs: Additional execution parameters

        Returns:
            SkillExecutionResult containing the execution result
        """
        if not self._initialized:
            self.discover_skills()

        skill = self.registry.get_skill(skill_id)
        if not skill:
            return SkillExecutionResult(
                success=False,
                output=None,
                error=f"Skill '{skill_id}' not found"
            )

        if not skill.enabled:
            return SkillExecutionResult(
                success=False,
                output=None,
                error=f"Skill '{skill_id}' is disabled"
            )

        # Validate input
        if not skill.validate_input(input_data):
            return SkillExecutionResult(
                success=False,
                output=None,
                error=f"Invalid input for skill '{skill_id}'"
            )

        # Execute skill
        try:
            result = skill.execute(input_data, **kwargs)
            logger.info(f"Executed skill '{skill_id}' successfully: {result.success}")
            return result

        except Exception as e:
            logger.error(f"Unexpected error executing skill '{skill_id}': {e}")
            return SkillExecutionResult(
                success=False,
                output=None,
                error=f"Execution error: {str(e)}"
            )

    def get_skill_info(self, skill_id: str) -> Optional[dict[str, Any]]:
        """
        Get detailed information about a skill.

        Args:
            skill_id: ID of the skill

        Returns:
            Dictionary with skill information or None if not found
        """
        skill = self.registry.get_skill(skill_id)
        if not skill:
            return None

        return skill.get_schema()

    def list_skills(self, category: Optional[str] = None, enabled_only: bool = True) -> list[dict[str, Any]]:
        """
        List available skills with their information.

        Args:
            category: Optional category filter
            enabled_only: Only return enabled skills

        Returns:
            List of skill information dictionaries
        """
        if not self._initialized:
            self.discover_skills()

        skills = self.registry.list_skills(category=category, enabled_only=enabled_only)
        return [skill.get_schema() for skill in skills]

    def search_skills(self, query: str) -> list[dict[str, Any]]:
        """
        Search for skills by query.

        Args:
            query: Search query

        Returns:
            List of matching skill information dictionaries
        """
        if not self._initialized:
            self.discover_skills()

        skills = self.registry.search_skills(query)
        return [skill.get_schema() for skill in skills]

    def get_categories(self) -> list[str]:
        """
        Get list of available skill categories.

        Returns:
            List of category names
        """
        if not self._initialized:
            self.discover_skills()

        return self.registry.list_categories()

    def enable_skill(self, skill_id: str) -> bool:
        """
        Enable a skill.

        Args:
            skill_id: ID of the skill to enable

        Returns:
            True if skill was enabled, False if not found
        """
        skill = self.registry.get_skill(skill_id)
        if not skill:
            return False

        skill.enable()
        logger.info(f"Enabled skill '{skill_id}'")
        return True

    def disable_skill(self, skill_id: str) -> bool:
        """
        Disable a skill.

        Args:
            skill_id: ID of the skill to disable

        Returns:
            True if skill was disabled, False if not found
        """
        skill = self.registry.get_skill(skill_id)
        if not skill:
            return False

        skill.disable()
        logger.info(f"Disabled skill '{skill_id}'")
        return True

    def register_custom_skill(self, skill: AgenticSkill) -> None:
        """
        Register a custom skill.

        Args:
            skill: The skill to register
        """
        self.registry.register(skill)
        logger.info(f"Registered custom skill '{skill.skill_id}'")

    def export_skills_config(self) -> str:
        """
        Export skills configuration as JSON.

        Returns:
            JSON string containing skills configuration
        """
        if not self._initialized:
            self.discover_skills()

        config = {
            "version": "1.0.0",
            "skills": self.registry.get_schema(),
            "categories": self.get_categories()
        }

        return json.dumps(config, indent=2, ensure_ascii=False)

    def get_skill_execution_stats(self) -> dict[str, Any]:
        """
        Get statistics about skill execution.

        Returns:
            Dictionary containing execution statistics
        """
        # Basic stats for now - could be extended with more detailed tracking
        total_skills = len(self.registry._skills)
        enabled_skills = len([s for s in self.registry._skills.values() if s.enabled])
        categories = len(self.registry._categories)

        return {
            "total_skills": total_skills,
            "enabled_skills": enabled_skills,
            "disabled_skills": total_skills - enabled_skills,
            "categories": categories,
            "category_breakdown": {
                cat: len(skills) for cat, skills in self.registry._categories.items()
            }
        }
