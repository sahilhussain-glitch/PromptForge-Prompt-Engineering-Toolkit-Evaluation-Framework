"""LLM runner — executes rendered prompts against OpenAI or Anthropic."""
import time
from openai import OpenAI
import anthropic

from config import get_settings

settings = get_settings()
_openai = OpenAI(api_key=settings.openai_api_key)
_anthropic = anthropic.Anthropic(api_key=settings.anthropic_api_key)


def run_prompt(rendered: dict, json_mode: bool = False, temperature: float = 0.1) -> dict:
    """
    Execute a rendered prompt (from PromptTemplate.render()).
    Returns {output, tokens_used, latency_ms, model}.
    """
    system = rendered["system"]
    user = rendered["user"]

    start = time.time()
    if settings.llm_provider == "anthropic":
        result = _run_anthropic(system, user)
    else:
        result = _run_openai(system, user, json_mode=json_mode, temperature=temperature)
    latency = int((time.time() - start) * 1000)

    return {
        "output": result["text"],
        "tokens_used": result["tokens"],
        "latency_ms": latency,
        "model": result["model"],
        "template_name": rendered.get("template_name"),
        "version": rendered.get("version"),
    }


def _run_openai(system: str, user: str, json_mode: bool, temperature: float) -> dict:
    kwargs = {
        "model": settings.chat_model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = _openai.chat.completions.create(**kwargs)
    return {
        "text": resp.choices[0].message.content.strip(),
        "tokens": resp.usage.total_tokens,
        "model": settings.chat_model,
    }


def _run_anthropic(system: str, user: str) -> dict:
    resp = _anthropic.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=0.1,
    )
    return {
        "text": resp.content[0].text.strip(),
        "tokens": resp.usage.input_tokens + resp.usage.output_tokens,
        "model": "claude-opus-4-5",
    }
