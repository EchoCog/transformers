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
Pipeline-based agentic skills implementation.

This module provides skills that wrap transformers pipelines, making them
available as agentic capabilities for chatbots and other AI systems.
"""

import logging
from typing import Any, Optional

from .base import AgenticSkill, SkillExecutionResult, SkillMetadata


try:
    from transformers import pipeline
    from transformers.pipelines import PIPELINE_REGISTRY
    _transformers_available = True
except ImportError:
    _transformers_available = False


logger = logging.getLogger(__name__)


class PipelineSkill(AgenticSkill):
    """
    An agentic skill that wraps a transformers pipeline.
    """

    def __init__(self, skill_id: str, pipeline_task: str, model: Optional[str] = None, **pipeline_kwargs):
        """
        Initialize a pipeline skill.

        Args:
            skill_id: Unique identifier for this skill
            pipeline_task: The pipeline task (e.g., 'text-classification', 'sentiment-analysis')
            model: Optional model name/path to use
            **pipeline_kwargs: Additional arguments for pipeline creation
        """
        if not _transformers_available:
            raise ImportError("transformers library is required for PipelineSkill")

        self.pipeline_task = pipeline_task
        self.model = model
        self.pipeline_kwargs = pipeline_kwargs
        self._pipeline = None

        # Create metadata based on pipeline task
        metadata = self._create_metadata_for_task(skill_id, pipeline_task)
        super().__init__(skill_id, metadata)

    def _create_metadata_for_task(self, skill_id: str, task: str) -> SkillMetadata:
        """Create metadata for a pipeline task."""

        # Pipeline task definitions with their metadata
        task_definitions = {
            "text-classification": {
                "name": "Text Classification",
                "description": "Classify text into predefined categories",
                "category": "text_analysis",
                "input_types": ["text"],
                "output_types": ["classification_scores"],
                "examples": [{"input": "I love this product!", "output": "POSITIVE"}]
            },
            "sentiment-analysis": {
                "name": "Sentiment Analysis",
                "description": "Analyze the sentiment of text (positive, negative, neutral)",
                "category": "text_analysis",
                "input_types": ["text"],
                "output_types": ["sentiment_scores"],
                "examples": [{"input": "This is amazing!", "output": "POSITIVE (0.98)"}]
            },
            "question-answering": {
                "name": "Question Answering",
                "description": "Answer questions based on provided context",
                "category": "text_understanding",
                "input_types": ["question", "context"],
                "output_types": ["answer"],
                "examples": [{"input": {"question": "What is AI?", "context": "AI is artificial intelligence..."}, "output": "artificial intelligence"}]
            },
            "text-generation": {
                "name": "Text Generation",
                "description": "Generate text based on a prompt",
                "category": "text_generation",
                "input_types": ["text"],
                "output_types": ["generated_text"],
                "examples": [{"input": "Once upon a time", "output": "Once upon a time, there was a brave knight..."}]
            },
            "text2text-generation": {
                "name": "Text-to-Text Generation",
                "description": "Transform text from one form to another (translation, summarization, etc.)",
                "category": "text_generation",
                "input_types": ["text"],
                "output_types": ["transformed_text"],
                "examples": [{"input": "Translate to French: Hello", "output": "Bonjour"}]
            },
            "summarization": {
                "name": "Text Summarization",
                "description": "Generate concise summaries of longer texts",
                "category": "text_generation",
                "input_types": ["text"],
                "output_types": ["summary"],
                "examples": [{"input": "Long article...", "output": "Brief summary..."}]
            },
            "translation": {
                "name": "Translation",
                "description": "Translate text between languages",
                "category": "text_generation",
                "input_types": ["text"],
                "output_types": ["translated_text"],
                "examples": [{"input": "Hello world", "output": "Hola mundo"}]
            },
            "fill-mask": {
                "name": "Fill Mask",
                "description": "Fill in masked tokens in text",
                "category": "text_completion",
                "input_types": ["text_with_mask"],
                "output_types": ["filled_text"],
                "examples": [{"input": "The weather is [MASK] today", "output": "The weather is beautiful today"}]
            },
            "token-classification": {
                "name": "Token Classification",
                "description": "Classify individual tokens in text (NER, POS tagging)",
                "category": "text_analysis",
                "input_types": ["text"],
                "output_types": ["token_labels"],
                "examples": [{"input": "John works at Google", "output": "John:PERSON, Google:ORG"}]
            },
            "feature-extraction": {
                "name": "Feature Extraction",
                "description": "Extract numerical features/embeddings from text",
                "category": "text_analysis",
                "input_types": ["text"],
                "output_types": ["embeddings"],
                "examples": [{"input": "Hello world", "output": "[0.1, 0.2, ...]"}]
            },
            "image-classification": {
                "name": "Image Classification",
                "description": "Classify images into predefined categories",
                "category": "image_analysis",
                "input_types": ["image"],
                "output_types": ["classification_scores"],
                "examples": [{"input": "image.jpg", "output": "cat (0.95)"}]
            },
            "object-detection": {
                "name": "Object Detection",
                "description": "Detect and locate objects in images",
                "category": "image_analysis",
                "input_types": ["image"],
                "output_types": ["detected_objects"],
                "examples": [{"input": "image.jpg", "output": "person (0.9) at [x,y,w,h]"}]
            },
            "image-segmentation": {
                "name": "Image Segmentation",
                "description": "Segment objects in images at pixel level",
                "category": "image_analysis",
                "input_types": ["image"],
                "output_types": ["segmentation_masks"],
                "examples": [{"input": "image.jpg", "output": "segmentation masks"}]
            },
            "image-to-text": {
                "name": "Image to Text",
                "description": "Generate text descriptions of images",
                "category": "multimodal",
                "input_types": ["image"],
                "output_types": ["text"],
                "examples": [{"input": "image.jpg", "output": "A cat sitting on a chair"}]
            },
            "automatic-speech-recognition": {
                "name": "Speech Recognition",
                "description": "Convert speech audio to text",
                "category": "audio_analysis",
                "input_types": ["audio"],
                "output_types": ["text"],
                "examples": [{"input": "audio.wav", "output": "Hello, how are you?"}]
            },
            "audio-classification": {
                "name": "Audio Classification",
                "description": "Classify audio into predefined categories",
                "category": "audio_analysis",
                "input_types": ["audio"],
                "output_types": ["classification_scores"],
                "examples": [{"input": "audio.wav", "output": "music (0.85)"}]
            },
            "text-to-audio": {
                "name": "Text to Speech",
                "description": "Convert text to speech audio",
                "category": "audio_generation",
                "input_types": ["text"],
                "output_types": ["audio"],
                "examples": [{"input": "Hello world", "output": "audio.wav"}]
            }
        }

        # Get task definition or create default
        task_def = task_definitions.get(task, {
            "name": task.replace("-", " ").title(),
            "description": f"Perform {task} task",
            "category": "general",
            "input_types": ["any"],
            "output_types": ["any"],
            "examples": []
        })

        return SkillMetadata(
            name=task_def["name"],
            description=task_def["description"],
            category=task_def["category"],
            input_types=task_def["input_types"],
            output_types=task_def["output_types"],
            parameters={
                "pipeline_task": task,
                "model": self.model,
                **self.pipeline_kwargs
            },
            examples=task_def["examples"],
            tags=[task, "pipeline", "transformers"]
        )

    def _get_pipeline(self):
        """Get or create the pipeline instance."""
        if self._pipeline is None:
            try:
                if self.model:
                    self._pipeline = pipeline(self.pipeline_task, model=self.model, **self.pipeline_kwargs)
                else:
                    self._pipeline = pipeline(self.pipeline_task, **self.pipeline_kwargs)
                logger.info(f"Created pipeline for task '{self.pipeline_task}'")
            except Exception as e:
                logger.error(f"Failed to create pipeline for task '{self.pipeline_task}': {e}")
                raise
        return self._pipeline

    def execute(self, input_data: Any, **kwargs) -> SkillExecutionResult:
        """
        Execute the pipeline skill.

        Args:
            input_data: Input data for the pipeline
            **kwargs: Additional parameters for pipeline execution

        Returns:
            SkillExecutionResult with the pipeline output
        """
        if not self.enabled:
            return SkillExecutionResult(
                success=False,
                output=None,
                error="Skill is disabled"
            )

        try:
            # Get pipeline instance
            pipe = self._get_pipeline()

            # Execute pipeline
            result = pipe(input_data, **kwargs)

            return SkillExecutionResult(
                success=True,
                output=result,
                metadata={
                    "pipeline_task": self.pipeline_task,
                    "model": self.model,
                    "input_type": type(input_data).__name__
                }
            )

        except Exception as e:
            logger.error(f"Error executing pipeline skill '{self.skill_id}': {e}")
            return SkillExecutionResult(
                success=False,
                output=None,
                error=str(e),
                metadata={
                    "pipeline_task": self.pipeline_task,
                    "model": self.model
                }
            )

    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input for the pipeline.

        Args:
            input_data: Input to validate

        Returns:
            True if input appears valid
        """
        # Basic validation - pipeline will handle detailed validation
        if input_data is None:
            return False

        # For question-answering, expect dict with 'question' and 'context'
        if self.pipeline_task == "question-answering":
            return isinstance(input_data, dict) and "question" in input_data and "context" in input_data

        # For most other tasks, accept strings, lists, or appropriate data types
        return True


def create_pipeline_skills(
    tasks: Optional[list[str]] = None,
    models: Optional[dict[str, str]] = None,
    **pipeline_kwargs
) -> list[PipelineSkill]:
    """
    Create pipeline skills for the specified tasks.

    Args:
        tasks: List of pipeline tasks to create skills for. If None, creates for common tasks.
        models: Optional mapping of task -> model name/path
        **pipeline_kwargs: Additional arguments for pipeline creation

    Returns:
        List of created PipelineSkill instances
    """
    if not _transformers_available:
        logger.warning("transformers library not available, cannot create pipeline skills")
        return []

    # Default tasks if none specified
    if tasks is None:
        tasks = [
            "sentiment-analysis",
            "text-classification",
            "question-answering",
            "text-generation",
            "summarization",
            "fill-mask",
            "token-classification",
            "feature-extraction"
        ]

    models = models or {}
    skills = []

    for task in tasks:
        try:
            skill_id = f"pipeline_{task}".replace("-", "_")
            model = models.get(task)

            skill = PipelineSkill(
                skill_id=skill_id,
                pipeline_task=task,
                model=model,
                **pipeline_kwargs
            )
            skills.append(skill)
            logger.info(f"Created pipeline skill for task '{task}'")

        except Exception as e:
            logger.warning(f"Failed to create pipeline skill for task '{task}': {e}")

    return skills


def get_available_pipeline_tasks() -> list[str]:
    """
    Get list of available pipeline tasks.

    Returns:
        List of available pipeline task names
    """
    if not _transformers_available:
        return []

    try:
        return list(PIPELINE_REGISTRY.supported_tasks.keys())
    except Exception:
        # Fallback to common tasks
        return [
            "sentiment-analysis",
            "text-classification",
            "question-answering",
            "text-generation",
            "text2text-generation",
            "summarization",
            "translation",
            "fill-mask",
            "token-classification",
            "feature-extraction",
            "image-classification",
            "object-detection",
            "image-segmentation",
            "image-to-text",
            "automatic-speech-recognition",
            "audio-classification",
            "text-to-audio"
        ]
