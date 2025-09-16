#!/usr/bin/env python3
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
Agentic Skills Demo

This example demonstrates how to use the agentic skills framework
to make transformers capabilities available to chatbots and AI agents.
"""

import json

from transformers.skills import SkillManager, ChatSkillIntegration


def main():
    print("🤖 Transformers Agentic Skills Demo")
    print("=" * 50)
    
    # Initialize the skill manager
    print("\n1. Initializing Skill Manager...")
    skill_manager = SkillManager(auto_discover=True)
    
    # List available skills
    print("\n2. Available Skills:")
    skills = skill_manager.list_skills()
    for skill in skills:
        print(f"   • {skill['name']} ({skill['skill_id']})")
        print(f"     Category: {skill['category']}")
        print(f"     Description: {skill['description']}")
        print()
    
    # Demonstrate skill execution
    print("3. Executing Skills Directly:")
    print("-" * 30)
    
    # Example 1: Text Classification
    if skill_manager.get_skill_info("pipeline_text_classification"):
        print("\n📊 Text Classification Example:")
        result = skill_manager.execute_skill(
            "pipeline_text_classification", 
            "I absolutely love this new product! It's amazing!"
        )
        
        if result.success:
            print(f"   Input: 'I absolutely love this new product! It's amazing!'")
            print(f"   Result: {result.output}")
        else:
            print(f"   Error: {result.error}")
    
    # Example 2: Question Answering
    if skill_manager.get_skill_info("pipeline_question_answering"):
        print("\n❓ Question Answering Example:")
        qa_input = {
            "question": "What is the agentic skills framework?",
            "context": "The agentic skills framework allows transformers library capabilities to be implemented as skills that can be provisioned to chatbots and other AI systems. It provides auto-discovery, execution, and management of various ML capabilities."
        }
        
        result = skill_manager.execute_skill("pipeline_question_answering", qa_input)
        
        if result.success:
            print(f"   Question: {qa_input['question']}")
            print(f"   Answer: {result.output}")
        else:
            print(f"   Error: {result.error}")
    
    # Demonstrate chat integration
    print("\n4. Chat Integration Demo:")
    print("-" * 30)
    
    chat_handler = ChatSkillIntegration(skill_manager)
    
    # Example chat messages with skill invocations
    chat_messages = [
        "!skills",
        '!skill text-classification "This movie is terrible!"',
        '!skill fill-mask "The weather is [MASK] today"',
        "Hello, how are you?",  # Regular message
        '!skill question-answering {"question": "What is AI?", "context": "Artificial Intelligence (AI) is the simulation of human intelligence in machines."}'
    ]
    
    for message in chat_messages:
        print(f"\n💬 User: {message}")
        
        response = chat_handler.handle_chat_message(message)
        
        if response["is_skill"]:
            print(f"🤖 Bot: {response['response']}")
        else:
            print("🤖 Bot: (This would be handled by regular chat logic)")
    
    # Demonstrate skill provisioning for chatbots
    print("\n5. Provisioning Skills for Chatbot:")
    print("-" * 40)
    
    provisioning_config = skill_manager.provision_skills_for_chatbot()
    
    print(f"📦 Provisioned {provisioning_config['total_skills']} skills across {len(provisioning_config['categories'])} categories")
    print(f"📋 Categories: {', '.join(provisioning_config['categories'])}")
    
    # Save provisioning config to file
    with open("/tmp/chatbot_skills_config.json", "w") as f:
        json.dump(provisioning_config, f, indent=2)
    
    print("💾 Provisioning configuration saved to /tmp/chatbot_skills_config.json")
    
    # Show skill statistics
    print("\n6. Skill Statistics:")
    print("-" * 20)
    
    stats = skill_manager.get_skill_execution_stats()
    print(f"📊 Total Skills: {stats['total_skills']}")
    print(f"✅ Enabled: {stats['enabled_skills']}")
    print(f"❌ Disabled: {stats['disabled_skills']}")
    print(f"📁 Categories: {stats['categories']}")
    
    print("\n📚 Category Breakdown:")
    for category, count in stats['category_breakdown'].items():
        print(f"   • {category}: {count} skills")
    
    print("\n🎉 Demo completed! The agentic skills framework is ready for integration with chatbots.")
    print("\n💡 To use with your chatbot:")
    print("   1. Initialize SkillManager with auto_discover=True")
    print("   2. Use ChatSkillIntegration to handle skill invocations")
    print("   3. Parse messages for !skill commands")
    print("   4. Execute skills and format responses")


if __name__ == "__main__":
    main()