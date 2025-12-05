# Scenario-Aware JSON Re-Contextualization

A LangGraph-based agentic workflow that transforms JSON documents to match a new scenario while preserving structure and locked fields.

## What It Does

Takes an input JSON with multiple scenarios and re-contextualizes **ALL** content to match a newly selected scenario - including company names, brands, emails, URLs, KPIs, and narrative text.

**Guarantees:**

- Output JSON has **identical structure** to input (same keys, nesting, array lengths)
- `scenarioOptions` field remains **byte-for-byte identical** (the only locked field)
- **Everything else** is rewritten for the new scenario
- Emails updated: `mark@harvestbowls.com` → `mark@trendwave.com`
- Zero residual brand name references from old scenario

## Performance

| Metric              | Value                                    |
| ------------------- | ---------------------------------------- |
| **Total Runtime**   | **~8 seconds**                          |
| **LLM Calls**       | 12 parallel batches                      |
| **Retries**         | 0 (typical)                              |
| **Texts Processed** | 172                                      |
| **Validation**      | Schema ✓, Locked Fields ✓, Consistency ✓ |

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Run re-contextualization (by scenario index)
python -m src.main --input POC_sim_D.json --scenario 3

# Run with scenario string
python -m src.main --input POC_sim_D.json --scenario "A mid-market fashion retailer..."

# List available scenarios
python -m src.main --input POC_sim_D.json --list-scenarios
```

## Output

```
output/
├── output.json           # Re-contextualized JSON
└── validation_report.json # Validation results + runtime stats
```

### Validation Report Structure

```json
{
  "schemaMatch": true,
  "lockedFieldsUntouched": true,
  "changedFieldPaths": [
    "topicWizardData.lessonInformation.lesson",
    "topicWizardData.workplaceScenario.background.organizationName",
    "topicWizardData.simulationFlow[0].children[1].data.email.body",
    "... (172 paths total)"
  ],
  "scenarioConsistency": "OK",
  "failedPaths": [],
  "runtimeStats": {
    "latency_ms": 10692,
    "numRetries": 0,
    "numLLMCalls": 12
  }
}
```

---

## Architecture

### The Simple Approach

```
┌──────────────────────────────────────────────────────────────────────┐
│  1. EXTRACT                                                          │
│     • Pull ALL strings from JSON (no filtering)                      │
│     • Only skip: scenarioOptions (locked field)                      │
│     • Result: 172 text values with their JSON paths                  │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  2. SEND TO LLM                                                      │
│     • "Here's OLD scenario, here's NEW scenario"                     │
│     • "Rewrite these texts - change brands, emails, metrics, etc."   │
│     • LLM decides what needs changing based on context               │
│     • 12 parallel batches × 15 texts each                            │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  3. INJECT BACK                                                      │
│     • Deep copy original JSON structure                              │
│     • Replace each text at its original path                         │
│     • Restore locked scenarioOptions                                 │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  4. VALIDATE                                                         │
│     • Schema match? (same structure)                                 │
│     • Locked fields intact? (scenarioOptions unchanged)              │
│     • Brand leakage? (old brand names shouldn't appear)              │
│     • If leakage found → retry those specific texts                  │
└──────────────────────────────────────────────────────────────────────┘
```

### LangGraph Workflow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Parse/Extract  │────▶│    Generate     │────▶│    Assemble     │
│                 │     │  (parallel LLM) │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                        ┌─────────────────┐             │
                        │     Repair      │◀────────────┤ (if validation fails)
                        │  (retry loop)   │             │
                        └─────────────────┘             ▼
                                │               ┌─────────────────┐
                                └──────────────▶│    Validate     │────▶ END
                                                │                 │
                                                └─────────────────┘
```

### 5-Node Pipeline

| Node         | What It Does                                                     |
| ------------ | ---------------------------------------------------------------- |
| **Parse**    | Extract ALL strings from JSON, save `scenarioOptions` separately |
| **Generate** | Send texts to LLM in parallel batches, get rewritten versions    |
| **Assemble** | Put rewritten texts back into original JSON structure            |
| **Validate** | Check schema, locked fields, and brand name leakage              |
| **Repair**   | If validation fails, retry only the failed texts (max 3 times)   |

### Project Structure

```
context/
├── src/
│   ├── __init__.py      # Package marker
│   ├── config.py        # Model name, API settings, constants
│   ├── workflow.py      # LangGraph nodes + graph definition (~460 lines)
│   └── main.py          # CLI entry point
├── output/              # Generated outputs
├── requirements.txt
└── POC_sim_D.json       # Sample input
```

---

## Key Design Decisions

### 1. Send EVERYTHING to the LLM

**No filtering.** We don't skip URLs, emails, or any content. The LLM decides what's scenario-dependent:

- `mark@harvestbowls.com` → `mark@trendwave.com` ✓
- `https://example.com` → stays the same (generic)
- `Practice` → stays the same (not scenario-specific)

### 2. Only Lock `scenarioOptions`

Per requirements, only ONE field is locked:

```
scenarioOptions → MUST be byte-for-byte identical
Everything else → CAN change to match new scenario
```

### 3. Validate Brand Names Only

Leakage detection checks for **brand names**, not generic words:

```python
# Find brand patterns in old scenario
old_brands = ["HarvestBowls", "Nature's Crust"]  # CamelCase + possessives

# Check if any appear in output (except locked fields)
for brand in old_brands:
    if brand.lower() in output_text.lower():
        # LEAKAGE! Retry this text.
```

We DON'T flag generic words like "food", "menu", "team" even if they appear in old scenario.

### 4. Parallel Batching

- 172 texts ÷ 15 per batch = 12 batches
- All 12 batches run in parallel (asyncio)
- Total LLM time: ~10 seconds

### 5. Targeted Repair

If validation finds leakage:

1. Identify which specific texts failed
2. Reset only those texts
3. Re-generate only those texts
4. Loop back to validation

Max 3 retries, then output with warning.

---

## Validation Checks

| Check         | What It Verifies                  | Method                    |
| ------------- | --------------------------------- | ------------------------- |
| Schema Match  | Same keys, nesting, array lengths | Recursive comparison      |
| Locked Fields | `scenarioOptions` unchanged       | Deep equality             |
| Brand Leakage | No old brand names in output      | Pattern matching + search |

### Brand Detection (the only regex used)

```python
# CamelCase brands: HarvestBowls, TrendWave
re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', scenario)

# Possessive brands: Nature's Crust
re.findall(r"[A-Z][a-z]+'s(?:\s+[A-Z][a-z]+)?", scenario)
```

---

## Requirements

- Python 3.10+
- OpenAI API key
- Dependencies: `langgraph`, `langchain`, `langchain-openai`, `pydantic`, `python-dotenv`

## Cost Estimate

Per re-contextualization run (~172 text values):

| Component     | Calculation     | Cost            |
| ------------- | --------------- | --------------- |
| Input tokens  | ~18K × $0.10/1M | $0.0018         |
| Output tokens | ~12K × $0.40/1M | $0.0048         |
| **Total**     |                 | **~$0.007/run** |

At scale: ~$7 per 1,000 documents processed.
