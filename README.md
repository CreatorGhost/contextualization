# Scenario-Aware JSON Re-Contextualization

A LangGraph-based agentic workflow that transforms JSON documents to match a new scenario while preserving structure and locked fields.

## What It Does

Takes an input JSON with multiple scenarios and re-contextualizes **ALL** content to match a newly selected scenario - including company names, brands, person names, emails, URLs, KPIs, pricing, and narrative text.

**Guarantees:**

- Output JSON has **identical structure** to input (same keys, nesting, array lengths)
- `scenarioOptions` field remains **byte-for-byte identical** (the only locked field)
- **Everything else** is rewritten for the new scenario
- Brands updated: `HarvestBowls` → `TrendWave`
- Person names updated: `Mark Caldwell` → `Sarah Chen`
- Emails updated: `mark@harvestbowls.com` → `sarah@trendwave.com`
- **Industry-adaptive pricing**: `$2-$3` (food) → `$25-$35` (fashion)
- Zero residual references from old scenario

## Performance

| Metric              | Value           |
| ------------------- | --------------- |
| **Total Runtime**   | **7-8 seconds** |
| **LLM Calls**       | ~42 calls       |
| **Texts Processed** | 172             |
| **Retries**         | 0-1 typically   |

### Key Features

| Feature                       | Description                                                           |
| ----------------------------- | --------------------------------------------------------------------- |
| **LLM Entity Extraction**     | Parallel extraction of brands, people, emails from both scenarios     |
| **Few-Shot Learning**         | Dynamic examples generated from scenario mapping                      |
| **Industry-Adaptive Pricing** | Automatic scaling of prices to fit target industry                    |
| **Token-Aware Batching**      | Batches sized by token count (~600 tokens/batch)                      |
| **HTML Splitting**            | Large HTML split at `<h2>`, `<h3>` boundaries for parallel processing |
| **Comprehensive Validation**  | Checks brands, person names, email domains, industry consistency      |

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
    "... (91 paths total)"
  ],
  "scenarioConsistency": "OK",
  "failedPaths": [],
  "runtimeStats": {
    "latency_ms": 7616,
    "numRetries": 0,
    "numLLMCalls": 42
  }
}
```

---

## Architecture

### High-Level Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│  1. PARSE & EXTRACT                                                  │
│     • Pull ALL strings from JSON (172 text values)                   │
│     • Only skip: scenarioOptions (locked field)                      │
│     • LLM Entity Extraction (parallel):                              │
│       - OLD: company=HarvestBowls, competitor=Nature's Crust         │
│       - NEW: company=TrendWave, competitor=ChicStyles                │
│       - Person names, email domains, promotions                      │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  2. GENERATE (LLM Rewrite)                                           │
│     • Few-shot examples from entity mapping                          │
│     • Industry-adaptive pricing rules                                │
│     • 40 parallel batches × ~600 tokens = 5-6s                       │
│     • Large HTML split into sections, reassembled after              │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  3. ASSEMBLE                                                         │
│     • Deep copy original JSON structure                              │
│     • Inject rewritten texts at original paths                       │
│     • Reassemble split HTML sections                                 │
│     • Restore locked scenarioOptions                                 │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│  4. VALIDATE (LLM-Extracted Entities)                                │
│     • Schema match? (same structure)                                 │
│     • Locked fields intact? (scenarioOptions unchanged)              │
│     • No old entities? (brands, people, emails, promotions)          │
│     • If leakage found → targeted repair (max 3 retries)             │
└──────────────────────────────────────────────────────────────────────┘
```

### LangGraph Workflow

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│   Parse & Extract   │────▶│      Generate       │────▶│    Assemble     │
│  + LLM Entity Ext.  │     │  (parallel batches) │     │                 │
│    (parallel)       │     │  + Few-Shot Learn   │     │                 │
└─────────────────────┘     └─────────────────────┘     └─────────────────┘
                                                                │
                            ┌─────────────────┐                 │
                            │     Repair      │◀────────────────┤ (if validation fails)
                            │  (retry failed) │                 │
                            └─────────────────┘                 ▼
                                    │                   ┌─────────────────┐
                                    └──────────────────▶│    Validate     │────▶ END
                                                        │ (LLM entities)  │
                                                        └─────────────────┘
```

### 5-Node Pipeline

| Node         | What It Does                                                         |
| ------------ | -------------------------------------------------------------------- |
| **Parse**    | Extract ALL strings, LLM entity extraction (parallel for old/new)    |
| **Generate** | Rewrite with few-shot examples + industry-adaptive pricing           |
| **Assemble** | Inject texts back, reassemble HTML sections, restore locked fields   |
| **Validate** | Check schema, locked fields, entity leakage (brands, people, emails) |
| **Repair**   | Retry only failed texts with specific feedback (max 3 retries)       |

### Project Structure

```
context/
├── src/
│   ├── __init__.py      # Package marker
│   ├── config.py        # Model name, API settings, batching constants
│   ├── workflow.py      # LangGraph nodes + graph definition (~1000 lines)
│   └── main.py          # CLI entry point
├── output/              # Generated outputs
├── requirements.txt
└── POC_sim_D.json       # Sample input
```

---

## Key Design Decisions

### 1. LLM-Based Entity Extraction

Instead of brittle regex patterns, we use LLM to semantically extract entities from scenarios:

```python
# LLM extracts these entities (run in parallel for old & new scenarios):
{
  "primary_company": "HarvestBowls",      # → "TrendWave"
  "competitor": "Nature's Crust",          # → "ChicStyles"
  "promotion_type": "$1 value menu",       # → "Buy One Get One Free"
  "person_names": ["Mark Caldwell", "Emily Carter"],
  "email_domains": ["harvestbowls.com"]
}
```

**Benefits:**

- Handles apostrophes: `Nature's Crust` ✓
- Finds names not in scenario text (from JSON context)
- Infers email domains from company names
- No regex maintenance required

### 2. Few-Shot Learning

Dynamic examples generated from entity mapping teach the LLM the transformation pattern:

```
TRANSFORMATION EXAMPLES:
─────────────────────────
Brand: "HarvestBowls" → "TrendWave"
Competitor: "Nature's Crust" → "ChicStyles"
Person: "Mark Caldwell" → [Generate new name like "Sarah Chen"]
Email: "mark@harvestbowls.com" → "sarah@trendwave.com"
Promotion: "$1 value menu" → "Buy One Get One Free promotion"
Industry: "fast food" → "fashion retail"
```

### 3. Industry-Adaptive Pricing

The LLM prompt includes rules to scale prices appropriately:

```
PRICING RULES:
• Food → Fashion: $2-$3 items → $25-$35 items
• Fashion → Food: $50-$70 items → $5-$7 items
• Scale ALL dollar amounts proportionally
• Keep percentages and relative comparisons the same
```

**Example Transformation:**

```
BEFORE (Food): "Launch a HarvestBites menu with items at $2-$3"
AFTER (Fashion): "Launch a TrendWave collection with items at $25-$35"
```

### 4. Comprehensive Validation

Validation now checks for ALL old scenario markers using LLM-extracted entities:

| Check                | What It Verifies                          |
| -------------------- | ----------------------------------------- |
| Schema Match         | Same keys, nesting, array lengths         |
| Locked Fields        | `scenarioOptions` byte-for-byte identical |
| Brand Leakage        | No old company/competitor names           |
| Person Name Leakage  | No old person names (LLM-extracted)       |
| Email Domain Leakage | No old email domains                      |
| Promotion Leakage    | No old promotion references               |

### 5. Only Lock `scenarioOptions`

Per requirements, only ONE field is locked:

```
scenarioOptions → MUST be byte-for-byte identical
Everything else → CAN change to match new scenario
```

### 6. Token-Aware Parallel Batching

| Parameter             | Value                              |
| --------------------- | ---------------------------------- |
| **Batch Size**        | Dynamic ~600 tokens per batch      |
| **Large Text (HTML)** | Split at `<h2>`, `<h3>` boundaries |
| **Parallelism**       | All batches run concurrently       |
| **Total Batches**     | ~40 batches                        |
| **Max Batch Time**    | ~5 seconds                         |

### 7. Targeted Repair

If validation finds leakage:

1. Identify which specific texts failed
2. Reset only those texts
3. Re-generate with specific feedback
4. Loop back to validation

Max 3 retries, then output with warning.

---

## Transformation Example

**Input Scenario (Food Industry):**

```
A strategy team at HarvestBowls is facing a drop in foot traffic after
Nature's Crust introduced a $1 value menu. As a business consultant,
learners must analyze the market shake-up...
```

**Output Scenario (Fashion Industry):**

```
A mid-market fashion retailer, TrendWave, is losing customers to
ChicStyles' aggressive Buy One Get One Free promotion. As a business
consultant, learners must analyze the market shake-up...
```

**What Changes:**

| Element        | Before                   | After                   |
| -------------- | ------------------------ | ----------------------- |
| Company        | HarvestBowls             | TrendWave               |
| Competitor     | Nature's Crust           | ChicStyles              |
| Promotion      | $1 value menu            | Buy One Get One Free    |
| Manager        | Mark Caldwell            | Sarah Chen              |
| Email          | mark@harvestbowls.com    | sarah@trendwave.com     |
| Pricing        | $2-$3 items              | $25-$35 items           |
| Industry Terms | fast food, menu, organic | fashion retail, apparel |

---

## Requirements

- Python 3.10+
- OpenAI API key
- Dependencies: `langgraph`, `langchain`, `langchain-openai`, `pydantic`, `python-dotenv`

## Cost Estimate

Per re-contextualization run (~172 text values):

| Component              | Calculation     | Cost            |
| ---------------------- | --------------- | --------------- |
| Entity extraction      | ~2K × $0.10/1M  | $0.0002         |
| Input tokens (rewrite) | ~18K × $0.10/1M | $0.0018         |
| Output tokens          | ~12K × $0.40/1M | $0.0048         |
| **Total**              |                 | **~$0.007/run** |

At scale: ~$7 per 1,000 documents processed.

---

## Verification Checklist

Run this to verify all requirements are met:

```bash
python -c "
import json
with open('output/output.json') as f: output = json.load(f)
with open('output/validation_report.json') as f: report = json.load(f)

print('1. Schema Match:', '✅' if report['schemaMatch'] else '❌')
print('2. Locked Fields:', '✅' if report['lockedFieldsUntouched'] else '❌')
print('3. Consistency:', '✅' if report['scenarioConsistency'] == 'OK' else '❌')
print('4. Runtime:', f\"{report['runtimeStats']['latency_ms']}ms\")
print('5. Changed Fields:', len(report['changedFieldPaths']))
"
```

Expected output:

```
1. Schema Match: ✅
2. Locked Fields: ✅
3. Consistency: ✅
4. Runtime: ~7600ms
5. Changed Fields: 91
```
