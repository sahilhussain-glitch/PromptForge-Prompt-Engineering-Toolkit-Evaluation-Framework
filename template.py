"""Prompt template engine with variable injection, few-shot builder, and CoT support."""
import re
import json
import random
from typing import Any
from dataclasses import dataclass, field


@dataclass
class FewShotExample:
    input: str
    output: str


@dataclass
class PromptTemplate:
    name: str
    system: str
    user_template: str
    few_shot_examples: list[dict] = field(default_factory=list)
    chain_of_thought: bool = False
    output_schema: dict = field(default_factory=dict)
    version: str = "v1"
    description: str = ""

    def render(self, variables: dict[str, Any], shuffle_examples: bool = False) -> dict:
        """
        Render the template with provided variables.
        Returns {"system": str, "user": str, "variables": dict}
        Raises ValueError for missing variables.
        """
        # Find required variables
        required = set(re.findall(r"\{\{(\w+)\}\}", self.user_template))
        missing = required - set(variables.keys())
        if missing:
            raise ValueError(f"Missing template variables: {missing}")

        # Inject variables
        user_content = self.user_template
        for key, value in variables.items():
            user_content = user_content.replace(f"{{{{{key}}}}}", str(value))

        # Build few-shot block
        examples = list(self.few_shot_examples)
        if shuffle_examples:
            random.shuffle(examples)

        system_content = self.system

        if examples:
            example_block = "\n\nExamples:\n"
            for ex in examples:
                inp = ex.get("input", "")
                out = ex.get("output", "")
                example_block += f"Input: {inp}\nOutput: {out}\n\n"
            system_content += example_block

        if self.output_schema:
            schema_str = json.dumps(self.output_schema, indent=2)
            system_content += f"\n\nExpected output schema (JSON):\n{schema_str}"

        if self.chain_of_thought:
            user_content += "\n\nThink step by step before giving your final answer."

        return {
            "system": system_content,
            "user": user_content,
            "variables": variables,
            "template_name": self.name,
            "version": self.version,
        }

    def count_tokens_estimate(self, variables: dict[str, Any]) -> int:
        """Rough token estimate (1 token ≈ 4 chars)."""
        rendered = self.render(variables)
        total_chars = len(rendered["system"]) + len(rendered["user"])
        return total_chars // 4

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "system": self.system,
            "user_template": self.user_template,
            "few_shot_examples": self.few_shot_examples,
            "chain_of_thought": self.chain_of_thought,
            "output_schema": self.output_schema,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PromptTemplate":
        return cls(**data)
