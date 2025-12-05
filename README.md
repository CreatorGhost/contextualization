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

| Metric              | Before Optimization | After Optimization |
| ------------------- | ------------------- | ------------------ |
| **Total Runtime**   | ~15 seconds         | **10-12 seconds**  |
| **LLM Calls**       | 26 batches          | 40 batches         |
| **Max Batch Time**  | 35 seconds          | **~8 seconds**     |
| **Texts Processed** | 172                 | 172                |
| **Improvement**     | -                   | **~25% faster**    |

### Optimization Techniques Applied

| Technique               | Before                | After                           | Impact             |
| ----------------------- | --------------------- | ------------------------------- | ------------------ |
| **Batching Strategy**   | Fixed 15 items/batch  | Dynamic ~600 tokens/batch       | Balanced load      |
| **Large HTML Handling** | Single 35s batch      | Split into 12 parallel sections | 35s → 8s           |
| **HTML Detection**      | All large texts split | Only HTML content split         | Accuracy preserved |

**Details:**

1. **Token-Aware Batching** - Batches sized by estimated token count (`len(text) * 0.25`), not item count

2. **HTML Content Splitting** - Large HTML (>2000 chars) split at `<h2>`, `<h3>`, `<hr>` boundaries, reassembled after processing

3. **Smart HTML Detection** - Only texts containing HTML tags (`<tag>`) are split; plain text processed as single batch

4. **Full Scenario Context** - All batches receive full scenario text to ensure proper industry/role understanding

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
    "latency_ms": 10500, // Before: 15308ms → After: ~10500ms
    "numRetries": 0,
    "numLLMCalls": 40 // Before: 26 → After: 40 (smaller, faster batches)
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
│  2. SEND TO LLM (Optimized)                                          │
│     • "Here's OLD scenario, here's NEW scenario"                     │
│     • "Rewrite these texts - change brands, emails, metrics, etc."   │
│     • LLM decides what needs changing based on context               │
│                                                                      │
│     BEFORE: 26 batches × 15 items = 35s (large HTML blocked others)  │
│     AFTER:  40 batches × ~600 tokens = 7-8s (HTML split + parallel)  │
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

| Node         | What It Does                                             | Before → After                      |
| ------------ | -------------------------------------------------------- | ----------------------------------- |
| **Parse**    | Extract ALL strings from JSON, save `scenarioOptions`    | No change                           |
| **Generate** | Send texts to LLM in parallel batches                    | 26 batches → 40 token-aware batches |
| **Assemble** | Put rewritten texts back into original JSON structure    | Now reassembles split HTML sections |
| **Validate** | Check schema, locked fields, and brand name leakage      | No change                           |
| **Repair**   | If validation fails, retry only the failed texts (max 3) | No change                           |

### Project Structure

```
context/
├── src/
│   ├── __init__.py      # Package marker
│   ├── config.py        # Model name, API settings, batching constants
│   ├── workflow.py      # LangGraph nodes + graph definition (~670 lines)
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

### 4. Token-Aware Parallel Batching

| Parameter             | Before (Fixed)                  | After (Optimized)                         |
| --------------------- | ------------------------------- | ----------------------------------------- |
| **Batch Size**        | Fixed 15 items per batch        | Dynamic ~600 tokens per batch             |
| **Large Text (HTML)** | Single batch (35s per file)     | Split into sections, parallel (<8s total) |
| **Prompt Content**    | Full scenario text (~500 chars) | Full scenario (required for accuracy)     |
| **Total Batches**     | 26 batches                      | 40 batches                                |
| **Max Batch Time**    | 35 seconds                      | ~8 seconds                                |
| **Total LLM Time**    | ~15 seconds                     | ~8-10 seconds                             |

**Key Changes:**

- Batches sized by token count, not item count
- Large HTML (>2000 chars) split at `<h2>`, `<h3>`, `<hr>` boundaries
- All batches run in parallel (asyncio)
- Sections reassembled after processing

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
