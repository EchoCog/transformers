# Agentic Skills Framework

The Agentic Skills Framework allows Transformers library capabilities to be implemented as skills that can be discovered, provisioned, and executed by chatbots and other AI systems. This enables seamless integration of various ML capabilities into conversational AI applications.

## Overview

The framework provides:

- **Auto-discovery** of available ML capabilities as skills
- **Standardized interface** for skill execution and management
- **Chat integration** for natural language skill invocation
- **Provisioning system** for deploying skills to chatbot systems
- **Metadata and documentation** for each skill

## Quick Start

### Basic Usage

```python
from transformers.skills import SkillManager

# Initialize with auto-discovery
skill_manager = SkillManager(auto_discover=True)

# List available skills
skills = skill_manager.list_skills()
for skill in skills:
    print(f"{skill['name']}: {skill['description']}")

# Execute a skill
result = skill_manager.execute_skill(
    "pipeline_text_classification", 
    "I love this product!"
)

if result.success:
    print(f"Result: {result.output}")
else:
    print(f"Error: {result.error}")
```

### Chat Integration

```python
from transformers.skills import ChatSkillIntegration

# Create chat handler with skills
chat_handler = ChatSkillIntegration()

# Handle skill invocations in chat
response = chat_handler.handle_chat_message('!skill text-classification "Amazing product!"')

if response["is_skill"]:
    print(response["response"])  # Formatted skill result
```

### Provisioning for Chatbots

```python
# Get configuration for chatbot deployment
config = skill_manager.provision_skills_for_chatbot(
    categories=["text_analysis", "text_generation"]
)

# Save configuration
import json
with open("chatbot_skills.json", "w") as f:
    json.dump(config, f, indent=2)
```

## CLI Usage

The framework includes a comprehensive CLI for managing skills:

### List Available Skills

```bash
transformers-cli skills list
```

### Search for Skills

```bash
transformers-cli skills search "sentiment"
```

### Execute a Skill

```bash
transformers-cli skills execute pipeline_text_classification "I love this!"
```

### Provision Skills for Chatbot

```bash
transformers-cli skills provision --output chatbot_config.json
```

### Get Skill Information

```bash
transformers-cli skills info pipeline_question_answering
```

## Skill Categories

Skills are organized into categories:

- **text_analysis**: Text classification, sentiment analysis, NER, feature extraction
- **text_understanding**: Question answering, reading comprehension
- **text_generation**: Text generation, summarization, translation
- **text_completion**: Fill mask, text completion
- **image_analysis**: Image classification, object detection
- **audio_analysis**: Speech recognition, audio classification
- **multimodal**: Image captioning, visual question answering

## Chat Commands

When using the chat integration, users can invoke skills with natural language commands:

### Basic Skill Invocation

```
!skill text-classification "This is amazing!"
!skill question-answering {"question": "What is AI?", "context": "AI is..."}
```

### Get Help

```
!skills
!skill help
```

### Skill Examples

```
# Text classification
!skill text-classification "I hate this product"

# Question answering
!skill question-answering {"question": "What is the capital of France?", "context": "France is a country in Europe. Its capital is Paris."}

# Fill mask
!skill fill-mask "The weather is [MASK] today"

# Token classification (NER)
!skill token-classification "John works at Google in California"
```

## API Reference

### SkillManager

The main interface for managing agentic skills.

```python
class SkillManager:
    def __init__(self, auto_discover: bool = True)
    def discover_skills(self) -> None
    def execute_skill(self, skill_id: str, input_data: Any, **kwargs) -> SkillExecutionResult
    def list_skills(self, category: Optional[str] = None, enabled_only: bool = True) -> List[Dict]
    def provision_skills_for_chatbot(self, categories: Optional[List[str]] = None) -> Dict[str, Any]
    def register_custom_skill(self, skill: AgenticSkill) -> None
    def enable_skill(self, skill_id: str) -> bool
    def disable_skill(self, skill_id: str) -> bool
```

### ChatSkillIntegration

Integration layer for chat interfaces.

```python
class ChatSkillIntegration:
    def __init__(self, skill_manager: Optional[SkillManager] = None)
    def handle_chat_message(self, message: str) -> Dict[str, Any]
    def is_skill_invocation(self, message: str) -> bool
    def execute_skill_from_message(self, message: str) -> Dict[str, Any]
    def get_available_skills_help(self) -> str
```

### SkillExecutionResult

Result object for skill execution.

```python
@dataclass
class SkillExecutionResult:
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]
    def to_json(self) -> str
```

## Custom Skills

You can create custom skills by extending the `AgenticSkill` base class:

```python
from transformers.skills import AgenticSkill, SkillMetadata, SkillExecutionResult

class CustomSkill(AgenticSkill):
    def __init__(self):
        metadata = SkillMetadata(
            name="Custom Skill",
            description="A custom skill implementation",
            category="custom",
            input_types=["text"],
            output_types=["result"],
            parameters={},
            examples=[{"input": "test", "output": "processed"}]
        )
        super().__init__("custom_skill", metadata)
    
    def execute(self, input_data, **kwargs):
        # Custom processing logic
        result = f"Processed: {input_data}"
        
        return SkillExecutionResult(
            success=True,
            output=result,
            metadata={"custom": True}
        )

# Register the custom skill
skill_manager.register_custom_skill(CustomSkill())
```

## Integration Examples

### Flask Chatbot

```python
from flask import Flask, request, jsonify
from transformers.skills import ChatSkillIntegration

app = Flask(__name__)
chat_handler = ChatSkillIntegration()

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    
    # Check for skill invocation
    response = chat_handler.handle_chat_message(message)
    
    if response["is_skill"]:
        return jsonify({"response": response["response"]})
    else:
        # Handle regular chat logic
        return jsonify({"response": "I'm a chatbot with ML skills!"})
```

### Discord Bot

```python
import discord
from transformers.skills import ChatSkillIntegration

client = discord.Client()
chat_handler = ChatSkillIntegration()

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    response = chat_handler.handle_chat_message(message.content)
    
    if response["is_skill"]:
        await message.channel.send(response["response"])
```

## Deployment

### Configuration File

Skills can be provisioned with a configuration file:

```json
{
  "skills": [
    {
      "id": "pipeline_text_classification",
      "name": "Text Classification",
      "description": "Classify text into categories",
      "category": "text_analysis",
      "input_types": ["text"],
      "output_types": ["classification_scores"],
      "enabled": true
    }
  ],
  "categories": ["text_analysis"],
  "total_skills": 1,
  "execution_endpoint": "execute_skill",
  "discovery_endpoint": "discover_skills"
}
```

### Environment Variables

```bash
# Disable auto-discovery
TRANSFORMERS_SKILLS_AUTO_DISCOVER=false

# Specify custom skills directory
TRANSFORMERS_SKILLS_PATH=/path/to/custom/skills
```

## Troubleshooting

### Common Issues

1. **No skills discovered**: Ensure PyTorch/TensorFlow is installed for pipeline skills
2. **Skill execution fails**: Check input format matches skill requirements
3. **Import errors**: Verify transformers installation includes skills module

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

skill_manager = SkillManager(auto_discover=True)
```

## Contributing

To add new skill types:

1. Create a new skill class extending `AgenticSkill`
2. Add discovery logic to `SkillManager._discover_skills()`
3. Update metadata definitions in skill classes
4. Add tests in `tests/skills/`
5. Update documentation

## License

The Agentic Skills Framework is part of the Transformers library and is licensed under the Apache License 2.0.