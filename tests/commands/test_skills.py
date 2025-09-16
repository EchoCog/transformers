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

import unittest
from argparse import Namespace
from unittest.mock import Mock, patch

from transformers.commands.skills import SkillsCommand


class TestSkillsCommand(unittest.TestCase):
    """Test SkillsCommand CLI functionality."""

    def setUp(self):
        self.mock_skill_manager = Mock()

    @patch('transformers.commands.skills.SkillManager')
    def test_list_skills_json_format(self, mock_manager_class):
        mock_manager_class.return_value = self.mock_skill_manager

        # Mock skills data
        mock_skills = [
            {
                "skill_id": "test_skill",
                "name": "Test Skill",
                "category": "test",
                "description": "A test skill",
                "enabled": True
            }
        ]
        self.mock_skill_manager.list_skills.return_value = mock_skills

        # Create command args
        args = Namespace(
            skills_action="list",
            category=None,
            enabled_only=True,
            all=False,
            format="json"
        )

        command = SkillsCommand(args)

        # Should not raise any exceptions
        try:
            command.run()
        except SystemExit:
            pass  # Expected for CLI commands

    @patch('transformers.commands.skills.SkillManager')
    def test_search_skills(self, mock_manager_class):
        mock_manager_class.return_value = self.mock_skill_manager

        # Mock search results
        mock_results = [
            {
                "skill_id": "sentiment_skill",
                "name": "Sentiment Analysis",
                "category": "text_analysis",
                "description": "Analyze sentiment",
                "enabled": True,
                "tags": ["sentiment", "analysis"]
            }
        ]
        self.mock_skill_manager.search_skills.return_value = mock_results

        args = Namespace(
            skills_action="search",
            query="sentiment",
            format="json"
        )

        command = SkillsCommand(args)

        try:
            command.run()
        except SystemExit:
            pass

        self.mock_skill_manager.search_skills.assert_called_once_with("sentiment")

    @patch('transformers.commands.skills.SkillManager')
    def test_execute_skill(self, mock_manager_class):
        mock_manager_class.return_value = self.mock_skill_manager

        # Mock execution result
        mock_result = Mock()
        mock_result.success = True
        mock_result.output = "Result output"
        mock_result.error = None
        mock_result.metadata = {"test": True}

        self.mock_skill_manager.execute_skill.return_value = mock_result

        args = Namespace(
            skills_action="execute",
            skill_id="test_skill",
            input="test input",
            params=None
        )

        command = SkillsCommand(args)

        try:
            command.run()
        except SystemExit:
            pass

        self.mock_skill_manager.execute_skill.assert_called_once_with("test_skill", "test input")

    @patch('transformers.commands.skills.SkillManager')
    def test_execute_skill_with_json_input(self, mock_manager_class):
        mock_manager_class.return_value = self.mock_skill_manager

        mock_result = Mock()
        mock_result.success = True
        mock_result.output = "Result"
        mock_result.error = None
        mock_result.metadata = None

        self.mock_skill_manager.execute_skill.return_value = mock_result

        args = Namespace(
            skills_action="execute",
            skill_id="qa_skill",
            input='{"question": "What is AI?", "context": "AI is artificial intelligence"}',
            params='{"max_length": 50}'
        )

        command = SkillsCommand(args)

        try:
            command.run()
        except SystemExit:
            pass

        # Should parse JSON input
        expected_input = {"question": "What is AI?", "context": "AI is artificial intelligence"}
        expected_params = {"max_length": 50}

        self.mock_skill_manager.execute_skill.assert_called_once_with(
            "qa_skill", expected_input, **expected_params
        )

    @patch('transformers.commands.skills.SkillManager')
    def test_provision_skills(self, mock_manager_class):
        mock_manager_class.return_value = self.mock_skill_manager

        # Mock provisioning config
        mock_config = {
            "skills": [
                {
                    "id": "test_skill",
                    "name": "Test Skill",
                    "category": "test",
                    "description": "A test skill"
                }
            ],
            "categories": ["test"],
            "total_skills": 1
        }
        self.mock_skill_manager.provision_skills_for_chatbot.return_value = mock_config

        args = Namespace(
            skills_action="provision",
            categories=["test"],
            output=None
        )

        command = SkillsCommand(args)

        try:
            command.run()
        except SystemExit:
            pass

        self.mock_skill_manager.provision_skills_for_chatbot.assert_called_once_with(categories=["test"])

    @patch('transformers.commands.skills.SkillManager')
    def test_enable_skill(self, mock_manager_class):
        mock_manager_class.return_value = self.mock_skill_manager
        self.mock_skill_manager.enable_skill.return_value = True

        args = Namespace(
            skills_action="enable",
            skill_id="test_skill"
        )

        command = SkillsCommand(args)

        try:
            command.run()
        except SystemExit:
            pass

        self.mock_skill_manager.enable_skill.assert_called_once_with("test_skill")

    @patch('transformers.commands.skills.SkillManager')
    def test_disable_skill(self, mock_manager_class):
        mock_manager_class.return_value = self.mock_skill_manager
        self.mock_skill_manager.disable_skill.return_value = True

        args = Namespace(
            skills_action="disable",
            skill_id="test_skill"
        )

        command = SkillsCommand(args)

        try:
            command.run()
        except SystemExit:
            pass

        self.mock_skill_manager.disable_skill.assert_called_once_with("test_skill")

    @patch('transformers.commands.skills.SkillManager')
    def test_show_stats(self, mock_manager_class):
        mock_manager_class.return_value = self.mock_skill_manager

        mock_stats = {
            "total_skills": 5,
            "enabled_skills": 4,
            "disabled_skills": 1,
            "categories": 3,
            "category_breakdown": {
                "text_analysis": 2,
                "text_generation": 2,
                "custom": 1
            }
        }
        self.mock_skill_manager.get_skill_execution_stats.return_value = mock_stats

        args = Namespace(skills_action="stats")

        command = SkillsCommand(args)

        try:
            command.run()
        except SystemExit:
            pass

        self.mock_skill_manager.get_skill_execution_stats.assert_called_once()

    @patch('transformers.commands.skills.SkillManager')
    def test_list_categories(self, mock_manager_class):
        mock_manager_class.return_value = self.mock_skill_manager

        mock_categories = ["text_analysis", "text_generation", "image_analysis"]
        self.mock_skill_manager.get_categories.return_value = mock_categories

        args = Namespace(skills_action="categories")

        command = SkillsCommand(args)

        try:
            command.run()
        except SystemExit:
            pass

        self.mock_skill_manager.get_categories.assert_called_once()


if __name__ == "__main__":
    unittest.main()
