# PromptForge — Prompt Engineering Toolkit & Evaluation Framework

> Design, version, test, and evaluate LLM prompts at scale. Built for engineers who care about reliability, accuracy, and reproducibility.

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green) ![Angular](https://img.shields.io/badge/Angular-18-red)

---

## What It Does

PromptForge is a developer toolkit for the full prompt lifecycle:

1. **Author** — Write prompts with dynamic variables, few-shot examples, and chain-of-thought activation
2. **Version** — Store and manage multiple versions of each prompt
3. **Test** — Run prompts against test cases and see structured outputs
4. **Evaluate** — Benchmark variants for accuracy, consistency, and token efficiency
5. **Document** — Auto-generate usage guides so non-engineers can use your prompts

---

## Architecture

```
Developer → Angular UI → FastAPI → Prompt Engine
                                        │
                         ┌──────────────┼─────────────────┐
                         ▼              ▼                  ▼
                   Template Engine   LLM Runner        Evaluator
                   (vars + few-shot) (OpenAI/Claude)   (accuracy/tokens)
                         │              │                  │
                         └──────────────▼──────────────────┘
                                  Prompt Store
                                (JSON file-based)
```

---

## Project Structure

```
promptforge/
├── engine/
│   ├── template.py        # Variable injection, few-shot builder, CoT toggle
│   ├── store.py           # Prompt versioning and storage (JSON)
│   └── runner.py          # LLM execution with multi-provider support
├── evaluator/
│   ├── benchmark.py       # Run prompt variants vs ground truth
│   └── metrics.py         # Accuracy, consistency, token efficiency metrics
├── api/
│   ├── main.py            # FastAPI entry
│   ├── routes_prompts.py  # CRUD for prompts
│   └── routes_eval.py     # Evaluation endpoints
├── frontend/
│   └── src/
│       ├── app/
│       │   ├── prompt-editor/
│       │   ├── evaluator/
│       │   └── results/
├── tests/
│   ├── test_template.py
│   └── test_evaluator.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/yourname/promptforge.git
cd promptforge
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn api.main:app --reload --port 8003
```

---

## Core Concepts

### Prompt Templates

Templates support variable injection `{{variable}}`, few-shot blocks, and CoT flags:

```python
from engine.template import PromptTemplate

pt = PromptTemplate(
    name="sentiment_classifier",
    system="You are a sentiment analysis expert. Respond only in JSON.",
    user_template="Classify the sentiment of: {{text}}",
    few_shot_examples=[
        {"input": "I love this!", "output": '{"sentiment": "positive", "score": 0.95}'},
        {"input": "This is awful.", "output": '{"sentiment": "negative", "score": -0.9}'},
    ],
    chain_of_thought=False,
    output_schema={"sentiment": "string", "score": "float"},
)

rendered = pt.render({"text": "The product is okay but shipping was slow."})
```

### Evaluation

```python
from evaluator.benchmark import run_benchmark

results = run_benchmark(
    prompt_name="sentiment_classifier",
    variants=["v1", "v2", "v3"],
    test_cases=[
        {"input": {"text": "Amazing!"}, "expected": {"sentiment": "positive"}},
        {"input": {"text": "Terrible!"}, "expected": {"sentiment": "negative"}},
    ]
)
# Returns accuracy, avg_tokens, consistency_score per variant
```

---

## Known LLM Limitations (Documented)

| Limitation | Mitigation in PromptForge |
|------------|---------------------------|
| Hallucination | JSON mode + output validation |
| Context window constraints | Token counter warns before limit |
| Positional bias | Few-shot examples shuffled per run |
| Inconsistency across runs | Consistency metric (run 5x, compare) |
| Verbose outputs | Explicit length constraints in system prompt |
