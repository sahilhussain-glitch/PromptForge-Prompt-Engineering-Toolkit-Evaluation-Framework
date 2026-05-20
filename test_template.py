"""Tests for the PromptForge template engine."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from engine.template import PromptTemplate


def make_template(**kwargs) -> PromptTemplate:
    defaults = dict(
        name="test_prompt", version="v1", system="You are a test assistant.",
        user_template="Answer: {{question}}", description="Test"
    )
    defaults.update(kwargs)
    return PromptTemplate(**defaults)


def test_render_basic():
    pt = make_template()
    result = pt.render({"question": "What is 2+2?"})
    assert "What is 2+2?" in result["user"]


def test_render_missing_variable():
    pt = make_template()
    with pytest.raises(ValueError, match="Missing template variables"):
        pt.render({})


def test_render_few_shot():
    pt = make_template(few_shot_examples=[
        {"input": "Hello", "output": "Hi there"}
    ])
    result = pt.render({"question": "test"})
    assert "Hello" in result["system"]
    assert "Hi there" in result["system"]


def test_chain_of_thought():
    pt = make_template(chain_of_thought=True)
    result = pt.render({"question": "test"})
    assert "step by step" in result["user"].lower()


def test_output_schema_injected():
    pt = make_template(output_schema={"sentiment": "string", "score": "float"})
    result = pt.render({"question": "test"})
    assert "sentiment" in result["system"]


def test_token_estimate():
    pt = make_template()
    count = pt.count_tokens_estimate({"question": "Hello world"})
    assert isinstance(count, int) and count > 0


def test_to_from_dict():
    pt = make_template()
    d = pt.to_dict()
    pt2 = PromptTemplate.from_dict(d)
    assert pt2.name == pt.name
    assert pt2.system == pt.system
