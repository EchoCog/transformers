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
Agentic Skills Framework for Transformers

This module provides a framework for implementing transformers library capabilities
as agentic skills that can be provisioned to chatbots and other AI systems.
"""

from .base import AgenticSkill, SkillExecutionResult, SkillMetadata, SkillRegistry
from .chat_integration import ChatSkillIntegration, create_skill_enabled_chat_handler
from .manager import SkillManager
from .pipeline_skills import PipelineSkill, create_pipeline_skills


__all__ = [
    "AgenticSkill",
    "SkillRegistry",
    "SkillExecutionResult",
    "SkillMetadata",
    "PipelineSkill",
    "create_pipeline_skills",
    "SkillManager",
    "ChatSkillIntegration",
    "create_skill_enabled_chat_handler",
]
