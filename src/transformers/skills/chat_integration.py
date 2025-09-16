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
Integration of agentic skills with chat interfaces.

This module provides utilities to integrate agentic skills with chatbot
systems and conversational AI interfaces.
"""

import json
import re
from typing import Any, Optional

from .manager import SkillManager


class ChatSkillIntegration:
    """
    Integration layer for using agentic skills in chat interfaces.

    This class provides utilities to:
    - Parse skill invocations from chat messages
    - Execute skills based on chat input
    - Format skill results for chat responses
    """

    def __init__(self, skill_manager: Optional[SkillManager] = None):
        """
        Initialize the chat skill integration.

        Args:
            skill_manager: Optional skill manager instance. If None, creates a new one.
        """
        self.skill_manager = skill_manager or SkillManager(auto_discover=True)

        # Pattern for detecting skill invocations in chat
        # Examples: !skill text-classification "I love this!", !use question-answering {...}
        self.skill_pattern = re.compile(
            r'!(skill|use)\s+([a-zA-Z_-]+)(?:\s+(.+))?',
            re.IGNORECASE | re.DOTALL
        )

    def parse_skill_invocation(self, message: str) -> Optional[tuple[str, Any, dict[str, Any]]]:
        """
        Parse a skill invocation from a chat message.

        Args:
            message: The chat message to parse

        Returns:
            Tuple of (skill_id, input_data, parameters) if found, None otherwise
        """
        match = self.skill_pattern.match(message.strip())
        if not match:
            return None

        command, skill_name, input_part = match.groups()

        # Convert skill name to skill ID format
        skill_id = f"pipeline_{skill_name}".replace("-", "_")

        if not input_part:
            return skill_id, None, {}

        input_part = input_part.strip()

        # Try to parse as JSON first
        try:
            parsed_input = json.loads(input_part)
            if isinstance(parsed_input, dict) and "input" in parsed_input:
                # Extract input and parameters
                input_data = parsed_input.pop("input")
                parameters = parsed_input
                return skill_id, input_data, parameters
            else:
                return skill_id, parsed_input, {}
        except json.JSONDecodeError:
            # If not JSON, treat as plain text
            # Remove quotes if present
            if input_part.startswith('"') and input_part.endswith('"'):
                input_part = input_part[1:-1]
            elif input_part.startswith("'") and input_part.endswith("'"):
                input_part = input_part[1:-1]

            return skill_id, input_part, {}

    def is_skill_invocation(self, message: str) -> bool:
        """
        Check if a message contains a skill invocation.

        Args:
            message: The message to check

        Returns:
            True if the message contains a skill invocation
        """
        return self.skill_pattern.match(message.strip()) is not None

    def execute_skill_from_message(self, message: str) -> dict[str, Any]:
        """
        Execute a skill based on a chat message.

        Args:
            message: The chat message containing skill invocation

        Returns:
            Dictionary with execution result and formatted response
        """
        parsed = self.parse_skill_invocation(message)
        if not parsed:
            return {
                "success": False,
                "error": "No valid skill invocation found",
                "response": "I couldn't find a valid skill command in your message. Use `!skill <skill-name> <input>` format."
            }

        skill_id, input_data, parameters = parsed

        # Execute the skill
        result = self.skill_manager.execute_skill(skill_id, input_data, **parameters)

        # Format response for chat
        if result.success:
            response = self._format_success_response(skill_id, result)
        else:
            response = self._format_error_response(skill_id, result)

        return {
            "success": result.success,
            "skill_id": skill_id,
            "input_data": input_data,
            "output": result.output,
            "error": result.error,
            "metadata": result.metadata,
            "response": response
        }

    def _format_success_response(self, skill_id: str, result) -> str:
        """Format a successful skill execution for chat response."""
        skill_info = self.skill_manager.get_skill_info(skill_id)
        skill_name = skill_info.get("name", skill_id) if skill_info else skill_id

        response = f"✅ **{skill_name}** executed successfully!\n\n"

        # Format output based on type
        output = result.output
        if isinstance(output, list):
            if len(output) == 1 and isinstance(output[0], dict):
                # Single result dictionary (common for classification, etc.)
                item = output[0]
                if "label" in item and "score" in item:
                    response += f"**Result:** {item['label']} (confidence: {item['score']:.3f})"
                else:
                    response += f"**Result:** {json.dumps(item, indent=2)}"
            else:
                # Multiple results
                response += "**Results:**\n"
                for i, item in enumerate(output[:3], 1):  # Limit to first 3 results
                    if isinstance(item, dict) and "label" in item and "score" in item:
                        response += f"{i}. {item['label']} (confidence: {item['score']:.3f})\n"
                    else:
                        response += f"{i}. {item}\n"
                if len(output) > 3:
                    response += f"... and {len(output) - 3} more results"
        elif isinstance(output, dict):
            if "answer" in output:
                # Question answering result
                response += f"**Answer:** {output['answer']}"
                if "score" in output:
                    response += f" (confidence: {output['score']:.3f})"
            else:
                response += f"**Result:** {json.dumps(output, indent=2)}"
        else:
            response += f"**Result:** {output}"

        return response

    def _format_error_response(self, skill_id: str, result) -> str:
        """Format an error response for chat."""
        skill_info = self.skill_manager.get_skill_info(skill_id)
        skill_name = skill_info.get("name", skill_id) if skill_info else skill_id

        response = f"❌ **{skill_name}** execution failed!\n\n"
        response += f"**Error:** {result.error}"

        return response

    def get_available_skills_help(self) -> str:
        """
        Get help text listing available skills for chat use.

        Returns:
            Formatted help text with available skills
        """
        skills = self.skill_manager.list_skills()

        if not skills:
            return "No skills are currently available."

        help_text = "**Available Agentic Skills:**\n\n"

        # Group by category
        by_category = {}
        for skill in skills:
            category = skill.get("category", "other")
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(skill)

        for category, category_skills in by_category.items():
            help_text += f"**{category.replace('_', ' ').title()}:**\n"

            for skill in category_skills:
                skill_name = skill["skill_id"].replace("pipeline_", "").replace("_", "-")
                description = skill["description"]
                examples = skill.get("examples", [])

                help_text += f"• `!skill {skill_name} <input>` - {description}\n"

                if examples:
                    example = examples[0]
                    if isinstance(example.get("input"), str):
                        help_text += f"  Example: `!skill {skill_name} \"{example['input']}\"`\n"
                    else:
                        help_text += f"  Example: `!skill {skill_name} {json.dumps(example['input'])}`\n"

            help_text += "\n"

        help_text += "**Usage:**\n"
        help_text += "• `!skill <skill-name> \"text input\"` - Execute skill with text\n"
        help_text += "• `!skill <skill-name> {\"key\": \"value\"}` - Execute skill with structured input\n"
        help_text += "• `!skills` - Show this help\n"

        return help_text

    def handle_chat_message(self, message: str) -> dict[str, Any]:
        """
        Handle a chat message, executing skills if found.

        Args:
            message: The chat message

        Returns:
            Dictionary with response information
        """
        message = message.strip()

        # Handle help request
        if message.lower() in ["!skills", "!skill help", "!help skills"]:
            return {
                "success": True,
                "is_skill": True,
                "response": self.get_available_skills_help()
            }

        # Check for skill invocation
        if self.is_skill_invocation(message):
            result = self.execute_skill_from_message(message)
            result["is_skill"] = True
            return result

        # Not a skill invocation
        return {
            "success": True,
            "is_skill": False,
            "response": None
        }


def create_skill_enabled_chat_handler(skill_manager: Optional[SkillManager] = None) -> ChatSkillIntegration:
    """
    Create a chat handler with skill capabilities.

    Args:
        skill_manager: Optional skill manager instance

    Returns:
        ChatSkillIntegration instance ready for use
    """
    return ChatSkillIntegration(skill_manager)
