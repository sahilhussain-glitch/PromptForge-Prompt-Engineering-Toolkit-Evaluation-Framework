"""PromptForge — FastAPI entry point."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__) + "/..")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Optional

from engine.template import PromptTemplate
from engine.store import save_prompt, load_prompt, list_prompts, list_versions, delete_prompt
from engine.runner import run_prompt

app = FastAPI(title="PromptForge API", version="1.0.0",
              description="Prompt Engineering Toolkit & Evaluation Framework")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Prompt CRUD ───────────────────────────────────────────────────────────────

class PromptCreate(BaseModel):
    name: str
    version: str = "v1"
    description: str = ""
    system: str
    user_template: str
    few_shot_examples: list[dict] = []
    chain_of_thought: bool = False
    output_schema: dict = {}


@app.post("/api/prompts")
def create_prompt(req: PromptCreate):
    pt = PromptTemplate(**req.model_dump())
    save_prompt(pt)
    return {"message": f"Prompt '{req.name}' v{req.version} saved", "prompt": pt.to_dict()}


@app.get("/api/prompts")
def get_prompts():
    return {"prompts": list_prompts()}


@app.get("/api/prompts/{name}/versions")
def get_versions(name: str):
    return {"name": name, "versions": list_versions(name)}


@app.get("/api/prompts/{name}/{version}")
def get_prompt(name: str, version: str):
    try:
        pt = load_prompt(name, version)
        return pt.to_dict()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Prompt not found")


@app.delete("/api/prompts/{name}/{version}")
def delete(name: str, version: str):
    if not delete_prompt(name, version):
        raise HTTPException(status_code=404, detail="Prompt not found")
    return {"message": f"Deleted {name} v{version}"}


# ── Run ───────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    name: str
    version: str = "v1"
    variables: dict[str, Any]
    json_mode: bool = False


@app.post("/api/run")
def run(req: RunRequest):
    try:
        pt = load_prompt(req.name, req.version)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Prompt not found")
    rendered = pt.render(req.variables)
    return run_prompt(rendered, json_mode=req.json_mode)


# ── Evaluate ──────────────────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    prompt_name: str
    versions: list[str]
    test_cases: list[dict]
    runs_per_case: int = 3


@app.post("/api/evaluate")
def evaluate(req: EvalRequest):
    from evaluator.benchmark import run_benchmark
    return run_benchmark(req.prompt_name, req.versions, req.test_cases, req.runs_per_case)


@app.get("/health")
def health():
    return {"status": "ok", "service": "PromptForge"}
