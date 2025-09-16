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
from unittest.mock import patch

from transformers.skills import (
    AgenticSkill,
    SkillExecutionResult,
    SkillManager,
    SkillMetadata,
    SkillRegistry,
)


class MockSkill(AgenticSkill):
    """Mock skill for testing."""

    def execute(self, input_data, **kwargs):
        return SkillExecutionResult(
            success=True,
            output=f"Processed: {input_data}",
            metadata={"mock": True}
        )


class TestSkillExecutionResult(unittest.TestCase):
    """Test SkillExecutionResult class."""

    def test_successful_result(self):
        result = SkillExecutionResult(
            success=True,
            output="test output",
            metadata={"key": "value"}
        )

        self.assertTrue(result.success)
        self.assertEqual(result.output, "test output")
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata, {"key": "value"})

    def test_failed_result(self):
        result = SkillExecutionResult(
            success=False,
            output=None,
            error="test error"
        )

        self.assertFalse(result.success)
        self.assertIsNone(result.output)
        self.assertEqual(result.error, "test error")

    def test_to_dict(self):
        result = SkillExecutionResult(
            success=True,
            output="test",
            metadata={"test": True}
        )

        expected = {
            "success": True,
            "output": "test",
            "error": None,
            "metadata": {"test": True}
        }

        self.assertEqual(result.to_dict(), expected)

    def test_to_json(self):
        result = SkillExecutionResult(success=True, output="test")
        json_str = result.to_json()

        self.assertIsInstance(json_str, str)
        self.assertIn('"success": true', json_str)
        self.assertIn('"output": "test"', json_str)


class TestSkillMetadata(unittest.TestCase):
    """Test SkillMetadata class."""

    def test_creation(self):
        metadata = SkillMetadata(
            name="Test Skill",
            description="A test skill",
            category="test",
            input_types=["text"],
            output_types=["text"],
            parameters={"param1": "value1"},
            examples=[{"input": "test", "output": "result"}]
        )

        self.assertEqual(metadata.name, "Test Skill")
        self.assertEqual(metadata.description, "A test skill")
        self.assertEqual(metadata.category, "test")
        self.assertEqual(metadata.input_types, ["text"])
        self.assertEqual(metadata.output_types, ["text"])
        self.assertEqual(metadata.tags, [])  # Default empty list
        self.assertEqual(metadata.version, "1.0.0")  # Default version


class TestAgenticSkill(unittest.TestCase):
    """Test AgenticSkill base class."""

    def setUp(self):
        self.metadata = SkillMetadata(
            name="Test Skill",
            description="A test skill",
            category="test",
            input_types=["text"],
            output_types=["text"],
            parameters={},
            examples=[]
        )
        self.skill = MockSkill("test_skill", self.metadata)

    def test_creation(self):
        self.assertEqual(self.skill.skill_id, "test_skill")
        self.assertEqual(self.skill.metadata.name, "Test Skill")
        self.assertTrue(self.skill.enabled)

    def test_execute(self):
        result = self.skill.execute("test input")

        self.assertTrue(result.success)
        self.assertEqual(result.output, "Processed: test input")
        self.assertEqual(result.metadata["mock"], True)

    def test_validate_input_default(self):
        # Default validation should always return True
        self.assertTrue(self.skill.validate_input("anything"))
        self.assertTrue(self.skill.validate_input(None))
        self.assertTrue(self.skill.validate_input(42))

    def test_enable_disable(self):
        self.assertTrue(self.skill.enabled)

        self.skill.disable()
        self.assertFalse(self.skill.enabled)

        self.skill.enable()
        self.assertTrue(self.skill.enabled)

    def test_get_schema(self):
        schema = self.skill.get_schema()

        expected_keys = [
            "skill_id", "name", "description", "category",
            "input_types", "output_types", "parameters",
            "examples", "tags", "version", "enabled"
        ]

        for key in expected_keys:
            self.assertIn(key, schema)

        self.assertEqual(schema["skill_id"], "test_skill")
        self.assertEqual(schema["name"], "Test Skill")
        self.assertTrue(schema["enabled"])


class TestSkillRegistry(unittest.TestCase):
    """Test SkillRegistry class."""

    def setUp(self):
        self.registry = SkillRegistry()

        metadata1 = SkillMetadata(
            name="Skill 1",
            description="First test skill",
            category="test",
            input_types=["text"],
            output_types=["text"],
            parameters={},
            examples=[],
            tags=["test", "first"]
        )
        self.skill1 = MockSkill("skill_1", metadata1)

        metadata2 = SkillMetadata(
            name="Skill 2",
            description="Second test skill",
            category="analysis",
            input_types=["text"],
            output_types=["analysis"],
            parameters={},
            examples=[],
            tags=["test", "second"]
        )
        self.skill2 = MockSkill("skill_2", metadata2)

    def test_register_skill(self):
        self.registry.register(self.skill1)

        retrieved_skill = self.registry.get_skill("skill_1")
        self.assertEqual(retrieved_skill, self.skill1)

    def test_register_duplicate_skill(self):
        self.registry.register(self.skill1)

        # Registering again should overwrite
        new_skill = MockSkill("skill_1", self.skill1.metadata)
        self.registry.register(new_skill)

        retrieved_skill = self.registry.get_skill("skill_1")
        self.assertEqual(retrieved_skill, new_skill)

    def test_unregister_skill(self):
        self.registry.register(self.skill1)

        success = self.registry.unregister("skill_1")
        self.assertTrue(success)

        retrieved_skill = self.registry.get_skill("skill_1")
        self.assertIsNone(retrieved_skill)

    def test_unregister_nonexistent_skill(self):
        success = self.registry.unregister("nonexistent")
        self.assertFalse(success)

    def test_list_skills(self):
        self.registry.register(self.skill1)
        self.registry.register(self.skill2)

        all_skills = self.registry.list_skills()
        self.assertEqual(len(all_skills), 2)

        test_skills = self.registry.list_skills(category="test")
        self.assertEqual(len(test_skills), 1)
        self.assertEqual(test_skills[0].skill_id, "skill_1")

    def test_list_categories(self):
        self.registry.register(self.skill1)
        self.registry.register(self.skill2)

        categories = self.registry.list_categories()
        self.assertIn("test", categories)
        self.assertIn("analysis", categories)

    def test_search_skills(self):
        self.registry.register(self.skill1)
        self.registry.register(self.skill2)

        # Search by name
        results = self.registry.search_skills("Skill 1")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].skill_id, "skill_1")

        # Search by description
        results = self.registry.search_skills("Second")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].skill_id, "skill_2")

        # Search by tag
        results = self.registry.search_skills("test")
        self.assertEqual(len(results), 2)  # Both have "test" tag

    def test_get_schema(self):
        self.registry.register(self.skill1)
        self.registry.register(self.skill2)

        schema = self.registry.get_schema()

        self.assertIn("skills", schema)
        self.assertIn("categories", schema)
        self.assertIn("total_skills", schema)
        self.assertIn("enabled_skills", schema)

        self.assertEqual(schema["total_skills"], 2)
        self.assertEqual(schema["enabled_skills"], 2)


class TestSkillManager(unittest.TestCase):
    """Test SkillManager class."""

    def setUp(self):
        # Create manager without auto-discovery to avoid dependency issues in tests
        self.manager = SkillManager(auto_discover=False)

    def test_creation_without_auto_discover(self):
        manager = SkillManager(auto_discover=False)
        self.assertIsNotNone(manager.registry)
        self.assertFalse(manager._initialized)

    @patch('transformers.skills.manager.create_pipeline_skills')
    @patch('transformers.skills.manager.get_available_pipeline_tasks')
    def test_discover_skills(self, mock_get_tasks, mock_create_skills):
        # Mock the pipeline discovery
        mock_get_tasks.return_value = ["sentiment-analysis", "text-classification"]

        metadata = SkillMetadata(
            name="Test Pipeline Skill",
            description="A test pipeline skill",
            category="test",
            input_types=["text"],
            output_types=["classification"],
            parameters={},
            examples=[]
        )
        mock_skill = MockSkill("pipeline_sentiment_analysis", metadata)
        mock_create_skills.return_value = [mock_skill]

        self.manager.discover_skills()

        self.assertTrue(self.manager._initialized)
        mock_create_skills.assert_called_once()

    def test_register_custom_skill(self):
        metadata = SkillMetadata(
            name="Custom Skill",
            description="A custom test skill",
            category="custom",
            input_types=["text"],
            output_types=["text"],
            parameters={},
            examples=[]
        )
        custom_skill = MockSkill("custom_skill", metadata)

        self.manager.register_custom_skill(custom_skill)

        skill_info = self.manager.get_skill_info("custom_skill")
        self.assertIsNotNone(skill_info)
        self.assertEqual(skill_info["name"], "Custom Skill")

    def test_execute_skill(self):
        metadata = SkillMetadata(
            name="Test Skill",
            description="A test skill",
            category="test",
            input_types=["text"],
            output_types=["text"],
            parameters={},
            examples=[]
        )
        test_skill = MockSkill("test_skill", metadata)
        self.manager.register_custom_skill(test_skill)

        result = self.manager.execute_skill("test_skill", "test input")

        self.assertTrue(result.success)
        self.assertEqual(result.output, "Processed: test input")

    def test_execute_nonexistent_skill(self):
        result = self.manager.execute_skill("nonexistent", "test input")

        self.assertFalse(result.success)
        self.assertIn("not found", result.error)

    def test_enable_disable_skill(self):
        metadata = SkillMetadata(
            name="Test Skill",
            description="A test skill",
            category="test",
            input_types=["text"],
            output_types=["text"],
            parameters={},
            examples=[]
        )
        test_skill = MockSkill("test_skill", metadata)
        self.manager.register_custom_skill(test_skill)

        # Test disable
        success = self.manager.disable_skill("test_skill")
        self.assertTrue(success)

        result = self.manager.execute_skill("test_skill", "test input")
        self.assertFalse(result.success)
        self.assertIn("disabled", result.error)

        # Test enable
        success = self.manager.enable_skill("test_skill")
        self.assertTrue(success)

        result = self.manager.execute_skill("test_skill", "test input")
        self.assertTrue(result.success)

    def test_provision_skills_for_chatbot(self):
        metadata = SkillMetadata(
            name="Test Skill",
            description="A test skill",
            category="test",
            input_types=["text"],
            output_types=["text"],
            parameters={},
            examples=[]
        )
        test_skill = MockSkill("test_skill", metadata)
        self.manager.register_custom_skill(test_skill)

        provisioning_config = self.manager.provision_skills_for_chatbot()

        self.assertIn("skills", provisioning_config)
        self.assertIn("categories", provisioning_config)
        self.assertIn("total_skills", provisioning_config)
        self.assertEqual(provisioning_config["total_skills"], 1)

        skills = provisioning_config["skills"]
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills[0]["id"], "test_skill")
        self.assertEqual(skills[0]["name"], "Test Skill")

    def test_export_skills_config(self):
        metadata = SkillMetadata(
            name="Test Skill",
            description="A test skill",
            category="test",
            input_types=["text"],
            output_types=["text"],
            parameters={},
            examples=[]
        )
        test_skill = MockSkill("test_skill", metadata)
        self.manager.register_custom_skill(test_skill)

        config_json = self.manager.export_skills_config()

        self.assertIsInstance(config_json, str)
        self.assertIn("version", config_json)
        self.assertIn("skills", config_json)
        self.assertIn("test_skill", config_json)


if __name__ == "__main__":
    unittest.main()
