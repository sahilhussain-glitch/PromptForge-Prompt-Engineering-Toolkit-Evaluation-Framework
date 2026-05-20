"""Prompt store — file-based versioning for prompt templates."""
import json
import os
from pathlib import Path
from engine.template import PromptTemplate

STORE_DIR = Path("prompt_store")


def _ensure_dir():
    STORE_DIR.mkdir(exist_ok=True)


def save_prompt(pt: PromptTemplate) -> None:
    """Save a prompt template. Filename: {name}__{version}.json"""
    _ensure_dir()
    path = STORE_DIR / f"{pt.name}__{pt.version}.json"
    path.write_text(json.dumps(pt.to_dict(), indent=2))


def load_prompt(name: str, version: str = "v1") -> PromptTemplate:
    _ensure_dir()
    path = STORE_DIR / f"{name}__{version}.json"
    if not path.exists():
        raise FileNotFoundError(f"Prompt '{name}' version '{version}' not found")
    data = json.loads(path.read_text())
    return PromptTemplate.from_dict(data)


def list_prompts() -> list[dict]:
    _ensure_dir()
    prompts = []
    for f in STORE_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            prompts.append({"name": data["name"], "version": data["version"],
                            "description": data.get("description", "")})
        except Exception:
            pass
    return sorted(prompts, key=lambda x: (x["name"], x["version"]))


def list_versions(name: str) -> list[str]:
    _ensure_dir()
    return sorted([
        f.stem.split("__")[1]
        for f in STORE_DIR.glob(f"{name}__*.json")
    ])


def delete_prompt(name: str, version: str) -> bool:
    path = STORE_DIR / f"{name}__{version}.json"
    if path.exists():
        path.unlink()
        return True
    return False
