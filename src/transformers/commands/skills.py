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
CLI command for managing agentic skills.
"""

import json
import sys
from argparse import ArgumentParser, Namespace
from typing import TYPE_CHECKING, Any, Optional

from transformers.commands import BaseTransformersCLICommand
from transformers.utils import is_rich_available


if TYPE_CHECKING:
    from transformers.skills import SkillManager


if is_rich_available():
    from rich.console import Console
    from rich.json import JSON
    from rich.panel import Panel
    from rich.table import Table


def print_json_pretty(data: Any, console: Optional[Any] = None):
    """Print JSON data with pretty formatting."""
    if is_rich_available() and console:
        console.print(JSON.from_data(data))
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False))


class SkillsCommand(BaseTransformersCLICommand):
    """Command for managing agentic skills."""

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """Register the skills subcommand."""
        skills_parser = parser.add_parser("skills", help="Manage agentic skills for chatbots")
        skills_subparsers = skills_parser.add_subparsers(dest="skills_action", help="Skills action")

        # List skills
        list_parser = skills_subparsers.add_parser("list", help="List available skills")
        list_parser.add_argument(
            "--category",
            type=str,
            help="Filter by category"
        )
        list_parser.add_argument(
            "--enabled-only",
            action="store_true",
            default=True,
            help="Show only enabled skills"
        )
        list_parser.add_argument(
            "--all",
            action="store_true",
            help="Show all skills including disabled ones"
        )
        list_parser.add_argument(
            "--format",
            choices=["table", "json"],
            default="table",
            help="Output format"
        )

        # Search skills
        search_parser = skills_subparsers.add_parser("search", help="Search for skills")
        search_parser.add_argument("query", type=str, help="Search query")
        search_parser.add_argument(
            "--format",
            choices=["table", "json"],
            default="table",
            help="Output format"
        )

        # Show skill details
        info_parser = skills_subparsers.add_parser("info", help="Show detailed skill information")
        info_parser.add_argument("skill_id", type=str, help="Skill ID to show info for")

        # Execute skill
        exec_parser = skills_subparsers.add_parser("execute", help="Execute a skill")
        exec_parser.add_argument("skill_id", type=str, help="Skill ID to execute")
        exec_parser.add_argument("input", type=str, help="Input data (JSON string or plain text)")
        exec_parser.add_argument(
            "--params",
            type=str,
            help="Additional parameters as JSON string"
        )

        # Provision skills for chatbot
        provision_parser = skills_subparsers.add_parser("provision", help="Provision skills for chatbot")
        provision_parser.add_argument(
            "--categories",
            nargs="+",
            help="Categories to include (default: all)"
        )
        provision_parser.add_argument(
            "--output",
            type=str,
            help="Output file for provisioning config"
        )

        # Enable/disable skills
        enable_parser = skills_subparsers.add_parser("enable", help="Enable a skill")
        enable_parser.add_argument("skill_id", type=str, help="Skill ID to enable")

        disable_parser = skills_subparsers.add_parser("disable", help="Disable a skill")
        disable_parser.add_argument("skill_id", type=str, help="Skill ID to disable")

        # Show statistics
        skills_subparsers.add_parser("stats", help="Show skill statistics")

        # Categories
        skills_subparsers.add_parser("categories", help="List skill categories")

        skills_parser.set_defaults(func=SkillsCommand)

    def __init__(self, args: Namespace):
        """Initialize the skills command."""
        self.args = args
        if is_rich_available():
            self.console = Console()
        else:
            self.console = None

    def run(self):
        """Run the skills command."""
        # Import here to avoid issues if transformers components are not available
        try:
            from transformers.skills import SkillManager
        except ImportError as e:
            print(f"Error: Skills functionality not available: {e}")
            sys.exit(1)

        # Initialize skill manager
        skill_manager = SkillManager(auto_discover=True)

        # Execute the appropriate action
        action = getattr(self.args, "skills_action", None)

        if action == "list":
            self._list_skills(skill_manager)
        elif action == "search":
            self._search_skills(skill_manager)
        elif action == "info":
            self._show_skill_info(skill_manager)
        elif action == "execute":
            self._execute_skill(skill_manager)
        elif action == "provision":
            self._provision_skills(skill_manager)
        elif action == "enable":
            self._enable_skill(skill_manager)
        elif action == "disable":
            self._disable_skill(skill_manager)
        elif action == "stats":
            self._show_stats(skill_manager)
        elif action == "categories":
            self._list_categories(skill_manager)
        else:
            self._show_help()

    def _list_skills(self, skill_manager: "SkillManager"):
        """List available skills."""
        enabled_only = self.args.enabled_only and not self.args.all
        skills = skill_manager.list_skills(
            category=getattr(self.args, "category", None),
            enabled_only=enabled_only
        )

        if self.args.format == "json":
            print_json_pretty(skills, self.console)
            return

        if not skills:
            print("No skills found matching the criteria.")
            return

        if is_rich_available() and self.console:
            table = Table(title="Available Agentic Skills")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Name", style="magenta")
            table.add_column("Category", style="green")
            table.add_column("Description")
            table.add_column("Status", justify="center")

            for skill in skills:
                status = "✓" if skill["enabled"] else "✗"
                status_style = "green" if skill["enabled"] else "red"

                table.add_row(
                    skill["skill_id"],
                    skill["name"],
                    skill["category"],
                    skill["description"][:60] + "..." if len(skill["description"]) > 60 else skill["description"],
                    f"[{status_style}]{status}[/{status_style}]"
                )

            self.console.print(table)
        else:
            # Fallback table format
            print(f"{'ID':<20} {'Name':<25} {'Category':<15} {'Status':<8} Description")
            print("-" * 100)
            for skill in skills:
                status = "enabled" if skill["enabled"] else "disabled"
                desc = skill["description"][:40] + "..." if len(skill["description"]) > 40 else skill["description"]
                print(f"{skill['skill_id']:<20} {skill['name']:<25} {skill['category']:<15} {status:<8} {desc}")

    def _search_skills(self, skill_manager: "SkillManager"):
        """Search for skills."""
        skills = skill_manager.search_skills(self.args.query)

        if self.args.format == "json":
            print_json_pretty(skills, self.console)
            return

        if not skills:
            print(f"No skills found matching query: '{self.args.query}'")
            return

        print(f"Found {len(skills)} skill(s) matching '{self.args.query}':")
        print()

        for skill in skills:
            if is_rich_available() and self.console:
                panel = Panel(
                    f"[bold]{skill['name']}[/bold]\n"
                    f"Category: [green]{skill['category']}[/green]\n"
                    f"Status: [{'green' if skill['enabled'] else 'red'}]{'Enabled' if skill['enabled'] else 'Disabled'}[/]\n"
                    f"Description: {skill['description']}\n"
                    f"Tags: {', '.join(skill['tags'])}",
                    title=f"[cyan]{skill['skill_id']}[/cyan]",
                    border_style="blue"
                )
                self.console.print(panel)
            else:
                print(f"ID: {skill['skill_id']}")
                print(f"Name: {skill['name']}")
                print(f"Category: {skill['category']}")
                print(f"Status: {'Enabled' if skill['enabled'] else 'Disabled'}")
                print(f"Description: {skill['description']}")
                print(f"Tags: {', '.join(skill['tags'])}")
                print("-" * 50)

    def _show_skill_info(self, skill_manager: "SkillManager"):
        """Show detailed skill information."""
        skill_info = skill_manager.get_skill_info(self.args.skill_id)

        if not skill_info:
            print(f"Skill '{self.args.skill_id}' not found.")
            return

        print_json_pretty(skill_info, self.console)

    def _execute_skill(self, skill_manager: "SkillManager"):
        """Execute a skill."""
        # Parse input data
        input_data = self.args.input
        try:
            # Try to parse as JSON first
            input_data = json.loads(input_data)
        except json.JSONDecodeError:
            # If not JSON, use as plain text
            pass

        # Parse additional parameters
        params = {}
        if getattr(self.args, "params", None):
            try:
                params = json.loads(self.args.params)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in parameters: {self.args.params}")
                return

        # Execute skill
        result = skill_manager.execute_skill(self.args.skill_id, input_data, **params)

        if is_rich_available() and self.console:
            if result.success:
                self.console.print(Panel(
                    f"[green]Success![/green]\n\nOutput:\n{json.dumps(result.output, indent=2, ensure_ascii=False)}",
                    title="Skill Execution Result",
                    border_style="green"
                ))
            else:
                self.console.print(Panel(
                    f"[red]Failed![/red]\n\nError: {result.error}",
                    title="Skill Execution Result",
                    border_style="red"
                ))
        else:
            if result.success:
                print("Execution successful!")
                print("Output:")
                print(json.dumps(result.output, indent=2, ensure_ascii=False))
            else:
                print("Execution failed!")
                print(f"Error: {result.error}")

        if result.metadata:
            print("\nMetadata:")
            print_json_pretty(result.metadata, self.console)

    def _provision_skills(self, skill_manager: "SkillManager"):
        """Provision skills for chatbot."""
        categories = getattr(self.args, "categories", None)
        provisioning_config = skill_manager.provision_skills_for_chatbot(categories=categories)

        if getattr(self.args, "output", None):
            with open(self.args.output, "w", encoding="utf-8") as f:
                json.dump(provisioning_config, f, indent=2, ensure_ascii=False)
            print(f"Provisioning configuration saved to: {self.args.output}")
        else:
            print_json_pretty(provisioning_config, self.console)

    def _enable_skill(self, skill_manager: "SkillManager"):
        """Enable a skill."""
        if skill_manager.enable_skill(self.args.skill_id):
            print(f"Skill '{self.args.skill_id}' enabled successfully.")
        else:
            print(f"Skill '{self.args.skill_id}' not found.")

    def _disable_skill(self, skill_manager: "SkillManager"):
        """Disable a skill."""
        if skill_manager.disable_skill(self.args.skill_id):
            print(f"Skill '{self.args.skill_id}' disabled successfully.")
        else:
            print(f"Skill '{self.args.skill_id}' not found.")

    def _show_stats(self, skill_manager: "SkillManager"):
        """Show skill statistics."""
        stats = skill_manager.get_skill_execution_stats()

        if is_rich_available() and self.console:
            self.console.print(Panel(
                f"Total Skills: [bold blue]{stats['total_skills']}[/bold blue]\n"
                f"Enabled: [green]{stats['enabled_skills']}[/green]\n"
                f"Disabled: [red]{stats['disabled_skills']}[/red]\n"
                f"Categories: [yellow]{stats['categories']}[/yellow]",
                title="Skill Statistics",
                border_style="blue"
            ))

            if stats['category_breakdown']:
                table = Table(title="Skills by Category")
                table.add_column("Category", style="cyan")
                table.add_column("Count", justify="right", style="magenta")

                for category, count in stats['category_breakdown'].items():
                    table.add_row(category, str(count))

                self.console.print(table)
        else:
            print("Skill Statistics:")
            print(f"  Total Skills: {stats['total_skills']}")
            print(f"  Enabled: {stats['enabled_skills']}")
            print(f"  Disabled: {stats['disabled_skills']}")
            print(f"  Categories: {stats['categories']}")

            if stats['category_breakdown']:
                print("\nSkills by Category:")
                for category, count in stats['category_breakdown'].items():
                    print(f"  {category}: {count}")

    def _list_categories(self, skill_manager: "SkillManager"):
        """List skill categories."""
        categories = skill_manager.get_categories()

        if is_rich_available() and self.console:
            table = Table(title="Skill Categories")
            table.add_column("Category", style="cyan")

            for category in sorted(categories):
                table.add_row(category)

            self.console.print(table)
        else:
            print("Available Categories:")
            for category in sorted(categories):
                print(f"  {category}")

    def _show_help(self):
        """Show help information."""
        print("Transformers Agentic Skills Management")
        print("")
        print("Available commands:")
        print("  list       - List available skills")
        print("  search     - Search for skills")
        print("  info       - Show detailed skill information")
        print("  execute    - Execute a skill")
        print("  provision  - Provision skills for chatbot")
        print("  enable     - Enable a skill")
        print("  disable    - Disable a skill")
        print("  stats      - Show skill statistics")
        print("  categories - List skill categories")
        print("")
        print("Use 'transformers-cli skills <command> --help' for detailed help on each command.")
