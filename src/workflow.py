"""
LangGraph workflow for JSON re-contextualization.
Simple approach: Extract ALL text, send to LLM, validate for leakage.
"""

import asyncio
import copy
import json
import logging
import re
import time
from typing import Any, Generator, TypedDict, Optional, Union, List, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from .config import (
    GENERATION_MODEL, MODEL_TEMPERATURE, OPENAI_API_KEY, MAX_RETRIES,
    TARGET_TOKENS_PER_BATCH, LARGE_TEXT_THRESHOLD, TOKENS_PER_CHAR
)

logger = logging.getLogger(__name__)

# =============================================================================
# STATE DEFINITION
# =============================================================================

class TextValue(TypedDict):
    path: str
    original: str
    rewritten: Optional[str]


class ValidationResult(TypedDict):
    schemaMatch: bool
    lockedFieldsUntouched: bool
    changedFieldPaths: List[str]
    scenarioConsistency: str
    failedPaths: List[str]
    runtimeStats: Dict[str, Any]


class ScenarioEntities(TypedDict):
    """Entities extracted from a scenario by LLM."""
    primary_company: str
    competitor: str
    promotion_type: str
    industry: str
    person_names: List[str]
    email_domains: List[str]
    key_terms: List[str]


class WorkflowState(TypedDict):
    input_json: Dict[str, Any]
    selected_scenario: Union[str, int]
    old_scenario: str
    new_scenario: str
    old_entities: Optional[ScenarioEntities]  # LLM-extracted from old scenario
    new_entities: Optional[ScenarioEntities]  # LLM-extracted from new scenario
    locked_fields: Dict[str, Any]
    text_values: List[TextValue]
    output_json: Optional[Dict[str, Any]]
    validation_result: Optional[ValidationResult]
    retry_count: int
    failed_paths: List[str]
    start_time: Optional[float]
    llm_call_count: int


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_all_strings(obj: Any, path: str = "") -> Generator[tuple[str, str], None, None]:
    """Extract ALL string values from JSON. No filtering - let LLM decide what to change."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            yield from extract_all_strings(value, new_path)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            yield from extract_all_strings(item, f"{path}[{idx}]")
    elif isinstance(obj, str) and len(obj) > 0:
        yield (path, obj)


def set_nested_value(obj: Any, path: str, value: Any) -> None:
    """Set value at nested path like 'topicWizardData.simulationFlow[0].name'."""
    parts = []
    current = ""
    i = 0
    while i < len(path):
        c = path[i]
        if c == ".":
            if current:
                parts.append(current)
                current = ""
        elif c == "[":
            if current:
                parts.append(current)
                current = ""
            j = i + 1
            while j < len(path) and path[j] != "]":
                j += 1
            parts.append(int(path[i+1:j]))
            i = j
        else:
            current += c
        i += 1
    if current:
        parts.append(current)
    
    current_obj = obj
    for part in parts[:-1]:
        current_obj = current_obj[part]
    current_obj[parts[-1]] = value


# =============================================================================
# LLM-BASED ENTITY EXTRACTION 
# =============================================================================

async def extract_entities_with_llm(scenario: str, json_sample: str = "") -> ScenarioEntities:
    """Use LLM to extract ALL scenario-dependent entities.
    
    Much more robust than regex - understands context and semantics.
    """
    llm = ChatOpenAI(model=GENERATION_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    
    prompt = f"""Analyze this business scenario and extract ALL scenario-specific entities.

SCENARIO:
{scenario}

{f"ADDITIONAL CONTEXT (sample from JSON):{chr(10)}{json_sample[:500]}" if json_sample else ""}

Extract and return as JSON:
{{
    "primary_company": "The main company facing the challenge (e.g., HarvestBowls, TrendWave)",
    "competitor": "The competing company (e.g., Nature's Crust, ChicStyles)",
    "promotion_type": "The specific promotion/challenge (e.g., $1 menu, Buy One Get One Free)",
    "industry": "The industry sector (e.g., fast-casual food, fashion retail, airlines)",
    "person_names": ["Any person names mentioned or implied"],
    "email_domains": ["Likely email domains based on company names, e.g., harvestbowls.com"],
    "key_terms": ["Industry-specific terms that would need to change, e.g., menu, eatery, collection, apparel"]
}}

Return ONLY valid JSON, no explanation."""

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    
    try:
        # Parse the JSON response
        content = response.content.strip()
        # Handle markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        entities = json.loads(content)
        
        return ScenarioEntities(
            primary_company=entities.get("primary_company", ""),
            competitor=entities.get("competitor", ""),
            promotion_type=entities.get("promotion_type", ""),
            industry=entities.get("industry", ""),
            person_names=entities.get("person_names", []),
            email_domains=entities.get("email_domains", []),
            key_terms=entities.get("key_terms", [])
        )
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM entity extraction response, falling back to empty")
        return ScenarioEntities(
            primary_company="",
            competitor="",
            promotion_type="",
            industry="",
            person_names=[],
            email_domains=[],
            key_terms=[]
        )


def get_old_markers_from_entities(old_entities: ScenarioEntities, new_entities: ScenarioEntities) -> List[str]:
    """Get markers from LLM-extracted entities that should NOT appear in output.
    
    Compares old vs new entities to find what should be replaced.
    """
    markers = []
    
    # Primary company (if different)
    if old_entities["primary_company"] and old_entities["primary_company"].lower() != new_entities["primary_company"].lower():
        markers.append(old_entities["primary_company"].lower())
        # Also add without spaces for CamelCase detection
        markers.append(old_entities["primary_company"].lower().replace(" ", ""))
    
    # Competitor
    if old_entities["competitor"] and old_entities["competitor"].lower() != new_entities["competitor"].lower():
        markers.append(old_entities["competitor"].lower())
        # Handle possessives like "Nature's"
        if "'" in old_entities["competitor"]:
            markers.append(old_entities["competitor"].split("'")[0].lower())
    
    # Promotion type
    if old_entities["promotion_type"] and old_entities["promotion_type"].lower() != new_entities["promotion_type"].lower():
        markers.append(old_entities["promotion_type"].lower())
    
    # Person names
    for name in old_entities["person_names"]:
        if name and name not in new_entities["person_names"]:
            markers.append(name.lower())
    
    # Email domains
    for domain in old_entities["email_domains"]:
        if domain and domain not in new_entities["email_domains"]:
            markers.append(domain.lower())
    
    # Filter out very short markers and duplicates
    markers = list(set(m for m in markers if len(m) > 2))
    
    return markers


# =============================================================================
# REGEX-BASED ENTITY EXTRACTION (Fallback)
# =============================================================================

def extract_scenario_brands(scenario: str) -> List[str]:
    """Extract brand/company names from a scenario.
    
    Looks for brand-like patterns: CamelCase, Possessives, Capitalized multi-word names.
    """
    brands = set()
    
    # 1. CamelCase words (HarvestBowls, TrendWave, ChicStyles)
    brands.update(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', scenario))
    
    # 2. Possessive phrases (Nature's Crust, Nature's)
    brands.update(re.findall(r"[A-Z][a-z]+'s(?:\s+[A-Z][a-z]+)?", scenario))
    
    # 3. Two-word capitalized names that look like brands (Fresh Taste, Blue Haven)
    brands.update(re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', scenario))
    
    return [b for b in brands if len(b) > 3]


def extract_scenario_elements(scenario: str) -> Dict[str, List[str]]:
    """Extract ALL scenario-dependent elements for comprehensive validation.
    
    Returns dict with: brands, promotions, industry_terms, key_phrases
    """
    elements = {
        "brands": [],
        "promotions": [],
        "industry_terms": [],
        "key_phrases": []
    }
    
    # 1. Brands (CamelCase, Possessives)
    elements["brands"].extend(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', scenario))
    elements["brands"].extend(re.findall(r"[A-Z][a-z]+'s(?:\s+[A-Z][a-z]+)?", scenario))
    
    # 2. Promotions (price-based patterns)
    elements["promotions"].extend(re.findall(r'\$\d+(?:\.\d+)?\s*(?:menu|meal|deal|value)', scenario, re.IGNORECASE))
    elements["promotions"].extend(re.findall(r'Buy One Get One Free', scenario, re.IGNORECASE))
    elements["promotions"].extend(re.findall(r'BOGO', scenario, re.IGNORECASE))
    elements["promotions"].extend(re.findall(r'\d+%[- ]off', scenario, re.IGNORECASE))
    
    # 3. Industry terms
    if any(term in scenario.lower() for term in ['food', 'restaurant', 'menu', 'fast-casual', 'eatery']):
        elements["industry_terms"].extend(['fast-casual', 'restaurant', 'menu', 'eatery', 'food brand'])
    if any(term in scenario.lower() for term in ['fashion', 'retail', 'apparel', 'clothing']):
        elements["industry_terms"].extend(['fashion', 'retailer', 'apparel', 'clothing', 'collection'])
    
    # 4. Key phrases (3+ word sequences with proper nouns)
    # Extract significant phrases containing brand names
    for brand in elements["brands"]:
        # Find phrases containing this brand
        pattern = rf'\b{re.escape(brand)}(?:\'s)?\s+\w+(?:\s+\w+)?'
        elements["key_phrases"].extend(re.findall(pattern, scenario))
    
    # Clean up - remove duplicates, filter short items
    for key in elements:
        elements[key] = list(set(e for e in elements[key] if len(str(e)) > 2))
    
    return elements


def get_old_scenario_markers(old_scenario: str, new_scenario: str) -> List[str]:
    """Get ALL markers from old scenario that should NOT appear in output.
    
    More comprehensive than just brand names - includes promotions, phrases, etc.
    """
    old_elements = extract_scenario_elements(old_scenario)
    new_elements = extract_scenario_elements(new_scenario)
    
    markers = []
    
    # Add brands unique to old scenario
    old_brands = set(b.lower() for b in old_elements["brands"])
    new_brands = set(b.lower() for b in new_elements["brands"])
    markers.extend(old_brands - new_brands)
    
    # Add promotions unique to old scenario  
    old_promos = set(p.lower() for p in old_elements["promotions"])
    new_promos = set(p.lower() for p in new_elements["promotions"])
    markers.extend(old_promos - new_promos)
    
    # Add key phrases
    markers.extend(p.lower() for p in old_elements["key_phrases"])
    
    return [m for m in markers if len(m) > 3]


def build_few_shot_examples(old_scenario: str, new_scenario: str) -> str:
    """Build few-shot transformation examples from the two scenarios.
    
    The model learns the mapping pattern from these examples.
    """
    old_elements = extract_scenario_elements(old_scenario)
    new_elements = extract_scenario_elements(new_scenario)
    
    examples = []
    
    # Brand mapping examples
    old_brands = old_elements["brands"][:2]  # Take first 2 brands
    new_brands = new_elements["brands"][:2]
    
    if old_brands and new_brands:
        # Primary company example
        examples.append(f'''Example 1 - Primary Company:
  BEFORE: "How should {old_brands[0]} respond to the competitive threat?"
  AFTER:  "How should {new_brands[0]} respond to the competitive threat?"''')
        
        # If we have competitors
        if len(old_brands) > 1 and len(new_brands) > 1:
            examples.append(f'''Example 2 - Competitor:
  BEFORE: "Analyze {old_brands[1]}'s strategy and its impact"
  AFTER:  "Analyze {new_brands[1]}'s strategy and its impact"''')
    
    # Promotion mapping example
    old_promos = old_elements["promotions"]
    new_promos = new_elements["promotions"]
    if old_promos and new_promos:
        examples.append(f'''Example 3 - Promotion/Challenge:
  BEFORE: "responding to the {old_promos[0]}"
  AFTER:  "responding to the {new_promos[0]}"''')
    
    # Email domain example
    if old_brands and new_brands:
        old_domain = old_brands[0].lower().replace("'s", "").replace(" ", "")
        new_domain = new_brands[0].lower().replace("'s", "").replace(" ", "")
        examples.append(f'''Example 4 - Email Domain:
  BEFORE: "contact@{old_domain}.com"
  AFTER:  "contact@{new_domain}.com"''')
    
    # Full sentence example
    if old_brands and new_brands:
        examples.append(f'''Example 5 - Full Sentence:
  BEFORE: "{old_brands[0]} must develop a strategic response to maintain market share"
  AFTER:  "{new_brands[0]} must develop a strategic response to maintain market share"''')
    
    # Person name example - IMPORTANT: Person names should also change for new context
    examples.append('''Example 6 - Person Names (Generate new appropriate names for the new scenario):
  BEFORE: "Mark Caldwell, Chief Strategy Officer"
  AFTER:  "Sarah Chen, Chief Strategy Officer" (or another appropriate name)
  
  BEFORE: "emily.carter@oldcompany.com"
  AFTER:  "emily.chen@newcompany.com"''')
    
    return "\n\n".join(examples) if examples else ""


def extract_json_person_names(input_json: Dict[str, Any]) -> List[str]:
    """Extract ACTUAL person names from the input JSON structure.
    
    Only looks for names in specific person-related fields like:
    - reportingManager.name
    - sender.name (in email contexts)
    - avatarName
    
    Avoids generic section/activity names.
    """
    names = set()
    
    def find_names(obj: Any, path: str = ""):
        if isinstance(obj, dict):
            # Only extract names from specific person-related contexts
            is_person_context = any(ctx in path.lower() for ctx in [
                "reportingmanager", "sender", "avatar", "manager"
            ])
            
            if is_person_context and "name" in obj and isinstance(obj["name"], str):
                name = obj["name"]
                # Person names typically have 2-3 words, first letter caps
                words = name.split()
                if 2 <= len(words) <= 4:
                    # Looks like a person name (First Last or First Middle Last)
                    if all(w[0].isupper() and w[1:].islower() for w in words if len(w) > 1):
                        names.add(name)
            
            # Also check for avatarName specifically
            if "avatarName" in obj and isinstance(obj["avatarName"], str):
                name = obj["avatarName"]
                words = name.split()
                if 2 <= len(words) <= 4:
                    names.add(name)
            
            for key, value in obj.items():
                find_names(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                find_names(item, f"{path}[{i}]")
    
    find_names(input_json)
    return list(names)


def compare_structure(obj1: Any, obj2: Any, path: str = "") -> List[str]:
    """Compare JSON structure, return list of differences."""
    diffs = []
    if type(obj1) != type(obj2):
        diffs.append(f"{path}: type mismatch")
        return diffs
    if isinstance(obj1, dict):
        for key in set(obj1.keys()) | set(obj2.keys()):
            if key not in obj1:
                diffs.append(f"{path}.{key}: extra in output")
            elif key not in obj2:
                diffs.append(f"{path}.{key}: missing in output")
            else:
                diffs.extend(compare_structure(obj1[key], obj2[key], f"{path}.{key}" if path else key))
    elif isinstance(obj1, list):
        if len(obj1) != len(obj2):
            diffs.append(f"{path}: array length mismatch")
        else:
            for i, (a, b) in enumerate(zip(obj1, obj2)):
                diffs.extend(compare_structure(a, b, f"{path}[{i}]"))
    return diffs


# =============================================================================
# NODE 1: PARSE & EXTRACT
# =============================================================================

def parse_extract_node(state: WorkflowState) -> Dict[str, Any]:
    """Extract ALL text values, save locked fields, and extract entities with LLM."""
    logger.info("=" * 60)
    logger.info("NODE: Parse & Extract")
    logger.info("=" * 60)
    start_time = time.time()
    
    input_json = state["input_json"]
    selected_scenario = state["selected_scenario"]
    topic_data = input_json.get("topicWizardData", {})
    
    # Save scenarioOptions (LOCKED - the ONLY locked field)
    scenario_options = topic_data.get("scenarioOptions", [])
    locked_fields = {"topicWizardData.scenarioOptions": scenario_options}
    logger.info(f"  LOCKED: scenarioOptions ({len(scenario_options)} items)")
    
    # Resolve old and new scenarios
    old_scenario = topic_data.get("selectedScenarioOption", "")
    if isinstance(selected_scenario, int):
        new_scenario = scenario_options[selected_scenario]
    else:
        new_scenario = selected_scenario
    
    # Extract ALL text values (only skip scenarioOptions - the locked field)
    text_values = [
        {"path": f"topicWizardData.{path}", "original": value, "rewritten": None}
        for path, value in extract_all_strings(topic_data)
        if "scenarioOptions" not in path
    ]
    
    logger.info(f"  Extracted {len(text_values)} text values")
    logger.info(f"  Old scenario: {old_scenario[:50]}...")
    logger.info(f"  New scenario: {new_scenario[:50]}...")
    
    # LLM-based entity extraction (run in parallel for both scenarios)
    logger.info("  Extracting entities with LLM (parallel)...")
    
    # Get a sample of JSON content to help LLM understand context
    json_sample = ""
    if topic_data.get("workplaceScenario"):
        json_sample = json.dumps(topic_data["workplaceScenario"], indent=2)[:500]
    
    async def extract_both():
        old_task = extract_entities_with_llm(old_scenario, json_sample)
        new_task = extract_entities_with_llm(new_scenario, "")
        return await asyncio.gather(old_task, new_task)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    old_entities, new_entities = loop.run_until_complete(extract_both())
    loop.close()
    
    logger.info(f"  OLD entities: company={old_entities['primary_company']}, competitor={old_entities['competitor']}")
    logger.info(f"  NEW entities: company={new_entities['primary_company']}, competitor={new_entities['competitor']}")
    
    return {
        "old_scenario": old_scenario,
        "new_scenario": new_scenario,
        "old_entities": old_entities,
        "new_entities": new_entities,
        "locked_fields": locked_fields,
        "text_values": text_values,
        "start_time": start_time,
        "llm_call_count": 2,  # 2 LLM calls for entity extraction
        "retry_count": 0,
        "failed_paths": [],
    }


# =============================================================================
# NODE 2: GENERATE (LLM REWRITE)
# =============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string."""
    return int(len(text) * TOKENS_PER_CHAR)


def split_html_into_sections(html: str) -> List[str]:
    """Split HTML content into logical sections for parallel processing.
    
    Splits on major block elements like <h2>, <h3>, <hr>, etc.
    Returns list of HTML sections that can be processed independently.
    """
    # Split on common section delimiters
    # Using regex to split while keeping the delimiters
    section_patterns = [
        r'(<h[1-3][^>]*>)',  # Headings h1-h3
        r'(<hr\s*/?>)',       # Horizontal rules
        r'(</ul>\s*<h)',      # End of list before heading
    ]
    
    combined_pattern = '|'.join(section_patterns)
    
    # First, try splitting on headings and hr tags
    parts = re.split(combined_pattern, html, flags=re.IGNORECASE)
    
    # Reassemble keeping delimiters with their following content
    sections = []
    current = ""
    
    for part in parts:
        if part is None:
            continue
        # Check if this is a delimiter
        if re.match(r'<h[1-3]|<hr|</ul>\s*<h', part, re.IGNORECASE):
            if current.strip():
                sections.append(current.strip())
            current = part
        else:
            current += part
    
    if current.strip():
        sections.append(current.strip())
    
    # If we couldn't split into meaningful sections, fall back to paragraph splits
    if len(sections) <= 1:
        # Split on paragraph boundaries
        para_parts = re.split(r'(</p>\s*<p)', html, flags=re.IGNORECASE)
        if len(para_parts) > 3:
            # Group every 3-5 paragraphs together
            sections = []
            current = ""
            para_count = 0
            for part in para_parts:
                current += part
                if '</p>' in part.lower():
                    para_count += 1
                if para_count >= 4:
                    sections.append(current)
                    current = ""
                    para_count = 0
            if current.strip():
                sections.append(current)
    
    # Filter out very small sections (pure HTML tags)
    sections = [s for s in sections if len(s.strip()) > 50]
    
    return sections if len(sections) > 1 else [html]


def create_token_aware_batches(texts: List[Dict]) -> tuple[List[List[Dict]], Dict[str, List[int]]]:
    """Create batches based on token count, not fixed size.
    
    - Large HTML texts (>LARGE_TEXT_THRESHOLD chars) are split into sections
    - Sections are processed in parallel, then reassembled
    - Other texts are grouped to stay under TARGET_TOKENS_PER_BATCH
    
    Returns:
        batches: List of text batches to process
        split_map: Dict mapping original path -> list of batch indices for reassembly
    """
    batches = []
    split_map = {}  # path -> [(batch_idx, section_idx)]
    
    large_texts = []
    small_texts = []
    
    for t in texts:
        if len(t["original"]) > LARGE_TEXT_THRESHOLD:
            large_texts.append(t)
        else:
            small_texts.append(t)
    
    # Split large HTML texts into sections and create individual batches
    # Only split texts that look like HTML (contain HTML tags)
    for t in large_texts:
        text = t["original"]
        is_html = bool(re.search(r'<[a-zA-Z][^>]*>', text))  # Has HTML tags
        
        if is_html:
            sections = split_html_into_sections(text)
        else:
            sections = [text]  # Don't split non-HTML, process as single batch
        
        if len(sections) > 1:
            # Track which batches belong to this text for reassembly
            split_map[t["path"]] = []
            for i, section in enumerate(sections):
                batch_idx = len(batches)
                split_map[t["path"]].append((batch_idx, i))
                # Create synthetic text dict for section
                batches.append([{
                    "path": f"{t['path']}__section_{i}",
                    "original": section,
                    "rewritten": None,
                    "_parent_path": t["path"],
                    "_section_idx": i
                }])
        else:
            # Couldn't split, process as single large batch
            batches.append([t])
    
    # Group small texts by token count
    current_batch = []
    current_tokens = 0
    
    for t in small_texts:
        text_tokens = estimate_tokens(t["original"])
        
        # If adding this text would exceed target, start new batch
        if current_tokens + text_tokens > TARGET_TOKENS_PER_BATCH and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        
        current_batch.append(t)
        current_tokens += text_tokens
    
    # Don't forget the last batch
    if current_batch:
        batches.append(current_batch)
    
    return batches, split_map


async def rewrite_batch(
    llm: ChatOpenAI, 
    batch: List[Dict], 
    old_scenario: str, 
    new_scenario: str, 
    few_shot_examples: str,
    batch_num: int,
) -> List[str]:
    """Rewrite a batch of texts using LLM with full scenario context and few-shot examples.
    
    Every batch gets the same comprehensive context to ensure consistency.
    """
    logger.info(f"      [Batch {batch_num}] Starting ({len(batch)} texts)...")
    start = time.time()
    
    # Format texts with numbers - FULL TEXT, NO SLICING
    text_list = "\n".join([f"[{i+1}] {t['original']}" for i, t in enumerate(batch)])
    
    # Comprehensive prompt with full scenarios and few-shot examples
    system_prompt = f"""You are re-contextualizing educational simulation content from one business scenario to another.

═══════════════════════════════════════════════════════════════════════════════
CURRENT SCENARIO (what the content currently reflects):
{old_scenario}

TARGET SCENARIO (what the content MUST reflect after your transformation):
{new_scenario}
═══════════════════════════════════════════════════════════════════════════════

TRANSFORMATION EXAMPLES - Learn the pattern:
────────────────────────────────────────────
{few_shot_examples}

═══════════════════════════════════════════════════════════════════════════════

CRITICAL RULES:
1. IDENTIFY corresponding elements between scenarios:
   - Primary company/brand (the one facing the challenge)
   - Competitor company/brand (the one causing the challenge)
   - The competitive challenge/promotion type
   - Industry context (food→fashion, airline→retail, etc.)
   - Person names, email domains, URLs

2. REPLACE ALL old scenario references with corresponding new ones:
   - Every brand mention → new brand
   - Every competitor mention → new competitor  
   - Every promotion/challenge reference → new promotion
   - Every email domain → new domain (e.g., @oldbrand.com → @newbrand.com)
   - Industry-specific terms → appropriate new industry terms
   - PERSON NAMES: Generate NEW appropriate names for the new scenario context
     (e.g., if original has "Mark Caldwell", create a new name like "Sarah Chen" or "Alex Rivera")

3. ADAPT PRICING & METRICS to be INDUSTRY-APPROPRIATE:
   - If changing from food to fashion: $2-$3 items → $25-$35 items, $7-$8 items → $70-$80 items
   - If changing from fashion to food: $50-$70 items → $5-$7 items
   - Scale ALL dollar amounts proportionally to fit the new industry's typical price points
   - Keep percentages (%), timeframes, and relative comparisons the same
   - Example: "menu items at $2-$3" in food → "apparel items at $25-$35" in fashion

4. BE 100% CONSISTENT:
   - Same entity = same replacement EVERYWHERE
   - If you change a person's name in one place, use the SAME new name everywhere
   - NEVER mix old and new references in the same text
   - If unsure, use the NEW scenario terminology

5. PRESERVE:
   - Structure, formatting (HTML/markdown intact)
   - Tone and style
   - Generic content that doesn't reference the scenario

6. OUTPUT FORMAT:
   - Return each text as [1], [2], etc. on its own line
   - Keep the FULL text, don't truncate

The output should read as if it was ORIGINALLY written for the target scenario.
NO TRACES of the old scenario should remain - including old person names and unrealistic pricing!"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"TEXTS TO TRANSFORM:\n{text_list}\n\nTransformed texts:"),
    ]
    
    response = await llm.ainvoke(messages)
    
    # Parse [N] prefixed results
    results = []
    current_text, current_num = "", 0
    for line in response.content.split("\n"):
        line = line.strip()
        if line.startswith("[") and "]" in line[:5]:
            if current_num > 0:
                results.append(current_text.strip())
            try:
                bracket_end = line.index("]")
                current_num = int(line[1:bracket_end])
                current_text = line[bracket_end+1:].strip()
            except:
                current_text += " " + line
        elif line:
            current_text += " " + line
    if current_num > 0:
        results.append(current_text.strip())
    
    # Pad with originals if LLM returned fewer results
    while len(results) < len(batch):
        results.append(batch[len(results)]["original"])
    
    logger.info(f"      [Batch {batch_num}] Done in {time.time()-start:.2f}s")
    return results[:len(batch)]


def generate_node(state: WorkflowState) -> Dict[str, Any]:
    """Rewrite text values in parallel batches with consistent context."""
    logger.info("=" * 60)
    logger.info("NODE: Generate (LLM Rewrite)")
    logger.info("=" * 60)
    
    text_values = state.get("text_values", [])
    failed_paths = state.get("failed_paths", [])
    llm_call_count = state.get("llm_call_count", 0)
    
    # Determine which texts to process
    if failed_paths:
        to_process = [t for t in text_values if t["path"] in failed_paths]
    else:
        to_process = [t for t in text_values if t["rewritten"] is None]
    
    if not to_process:
        return {"text_values": text_values, "llm_call_count": llm_call_count}
    
    # Create token-aware batches (large texts are split into sections)
    batches, split_map = create_token_aware_batches(to_process)
    section_count = sum(len(v) for v in split_map.values())
    logger.info(f"  Processing {len(to_process)} texts in {len(batches)} batches ({len(split_map)} texts split into {section_count} sections)")
    
    # Build few-shot examples from scenarios - SAME for ALL batches for consistency
    few_shot_examples = build_few_shot_examples(state["old_scenario"], state["new_scenario"])
    logger.info(f"  Generated few-shot examples for transformation")
    
    llm = ChatOpenAI(model=GENERATION_MODEL, temperature=MODEL_TEMPERATURE, api_key=OPENAI_API_KEY)
    
    async def run_all():
        tasks = []
        for i, b in enumerate(batches):
            # ALL batches get the SAME context (full scenarios + few-shot examples)
            # This ensures consistency across all transformations
            tasks.append(rewrite_batch(
                llm, b, 
                state["old_scenario"], state["new_scenario"],
                few_shot_examples,
                i+1
            ))
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    start = time.time()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(run_all())
    loop.close()
    logger.info(f"  LLM calls done in {time.time()-start:.2f}s")
    
    # Map results back to text_values
    result_map = {}
    section_results = {}  # path -> {section_idx: rewritten}
    
    for batch, batch_results in zip(batches, results):
        if not isinstance(batch_results, Exception):
            for text, rewritten in zip(batch, batch_results):
                if "_parent_path" in text:
                    # This is a section result, store for reassembly
                    parent = text["_parent_path"]
                    if parent not in section_results:
                        section_results[parent] = {}
                    section_results[parent][text["_section_idx"]] = rewritten
                else:
                    result_map[text["path"]] = rewritten
    
    # Reassemble split texts
    for path, sections in section_results.items():
        # Sort by section index and join
        sorted_sections = [sections[i] for i in sorted(sections.keys())]
        result_map[path] = "\n".join(sorted_sections)
    
    updated = [{**t, "rewritten": result_map.get(t["path"], t["rewritten"])} for t in text_values]
    
    return {"text_values": updated, "llm_call_count": llm_call_count + len(batches), "failed_paths": []}


# =============================================================================
# NODE 3: ASSEMBLE
# =============================================================================

def assemble_node(state: WorkflowState) -> Dict[str, Any]:
    """Reconstruct JSON with rewritten values."""
    logger.info("=" * 60)
    logger.info("NODE: Assemble")
    logger.info("=" * 60)
    
    output_json = copy.deepcopy(state["input_json"])
    
    # Inject rewritten texts
    count = 0
    for t in state.get("text_values", []):
        if t.get("rewritten"):
            try:
                set_nested_value(output_json, t["path"], t["rewritten"])
                count += 1
            except Exception as e:
                logger.warning(f"  Failed to set {t['path']}: {e}")
    logger.info(f"  Injected {count} rewritten values")
    
    # Restore locked scenarioOptions (MUST be byte-for-byte identical)
    output_json["topicWizardData"]["scenarioOptions"] = state["locked_fields"]["topicWizardData.scenarioOptions"]
    output_json["topicWizardData"]["selectedScenarioOption"] = state["new_scenario"]
    logger.info("  Restored locked scenarioOptions")
    
    return {"output_json": output_json}


# =============================================================================
# NODE 4: VALIDATE
# =============================================================================

def validate_node(state: WorkflowState) -> Dict[str, Any]:
    """Validate schema, locked fields, and check for old scenario leakage using LLM-extracted entities."""
    logger.info("=" * 60)
    logger.info("NODE: Validate")
    logger.info("=" * 60)
    
    input_json, output_json = state["input_json"], state["output_json"]
    
    # 1. Schema check
    schema_diffs = compare_structure(input_json, output_json)
    schema_ok = len(schema_diffs) == 0
    logger.info(f"  Schema: {'OK' if schema_ok else 'FAILED'}")
    
    # 2. Locked fields check
    locked_ok = (output_json["topicWizardData"]["scenarioOptions"] == 
                 state["locked_fields"]["topicWizardData.scenarioOptions"])
    logger.info(f"  Locked fields: {'OK' if locked_ok else 'FAILED'}")
    
    # 3. Comprehensive leakage check using LLM-extracted entities (more robust)
    old_entities = state.get("old_entities")
    new_entities = state.get("new_entities")
    
    if old_entities and new_entities:
        # Use LLM-extracted entities (more robust)
        old_markers = get_old_markers_from_entities(old_entities, new_entities)
        logger.info(f"  Using LLM-extracted entities for validation")
    else:
        # Fallback to regex-based extraction
        old_markers = get_old_scenario_markers(state["old_scenario"], state["new_scenario"])
        logger.info(f"  Using regex-based extraction for validation (fallback)")
    
    # Also add person names from original JSON (LLM-extracted or regex-based)
    if old_entities and old_entities.get("person_names"):
        for name in old_entities["person_names"]:
            if name and name.lower() not in old_markers:
                old_markers.append(name.lower())
    else:
        # Fallback: extract person names from JSON structure
        old_person_names = extract_json_person_names(state["input_json"])
        for name in old_person_names:
            name_lower = name.lower()
            if name_lower not in old_markers and len(name_lower) > 3:
                old_markers.append(name_lower)
    
    logger.info(f"  Checking for {len(old_markers)} old scenario markers: {old_markers[:8]}...")
    
    failed_paths = []
    failed_details = []  # Track what was found for better debugging
    
    for t in state.get("text_values", []):
        if "scenarioOptions" in t["path"]:
            continue  # Skip locked field
        if t.get("rewritten"):
            text_lower = t["rewritten"].lower()
            for marker in old_markers:
                if marker in text_lower:
                    failed_paths.append(t["path"])
                    failed_details.append(f"{t['path']}: found '{marker}'")
                    break
    
    if failed_details:
        logger.warning(f"  Leakage found in {len(failed_paths)} fields:")
        for detail in failed_details[:5]:  # Show first 5
            logger.warning(f"    - {detail}")
    
    consistency = "OK" if not failed_paths else f"FAILED: {len(failed_paths)} texts have old references"
    logger.info(f"  Consistency: {consistency}")
    
    # Build report
    latency_ms = int((time.time() - state.get("start_time", time.time())) * 1000)
    changed_paths = [t["path"] for t in state.get("text_values", []) 
                     if t.get("rewritten") and t["rewritten"] != t["original"]]
    
    return {
        "validation_result": {
            "schemaMatch": schema_ok,
            "lockedFieldsUntouched": locked_ok,
            "changedFieldPaths": changed_paths,
            "scenarioConsistency": consistency,
            "failedPaths": failed_paths,
            "runtimeStats": {
                "latency_ms": latency_ms, 
                "numRetries": state.get("retry_count", 0), 
                "numLLMCalls": state.get("llm_call_count", 0)
            },
        },
        "failed_paths": failed_paths,
    }


# =============================================================================
# NODE 5: REPAIR
# =============================================================================

def repair_node(state: WorkflowState) -> Dict[str, Any]:
    """Reset failed texts for re-generation."""
    logger.info("=" * 60)
    logger.info("NODE: Repair")
    logger.info("=" * 60)
    
    retry_count = state.get("retry_count", 0) + 1
    logger.info(f"  Retry {retry_count}/{MAX_RETRIES}")
    
    if retry_count > MAX_RETRIES:
        return {"retry_count": retry_count}
    
    # Reset failed texts so they get reprocessed
    failed = state.get("failed_paths", [])
    updated = [{**t, "rewritten": None} if t["path"] in failed else t 
               for t in state.get("text_values", [])]
    
    return {"text_values": updated, "retry_count": retry_count, "failed_paths": failed}


# =============================================================================
# GRAPH DEFINITION
# =============================================================================

def should_repair(state: WorkflowState) -> str:
    """Decide if repair is needed based on validation results."""
    result = state.get("validation_result", {})
    all_ok = (result.get("schemaMatch") and 
              result.get("lockedFieldsUntouched") and 
              result.get("scenarioConsistency", "").startswith("OK"))
    
    if all_ok:
        return "end"
    return "repair" if state.get("retry_count", 0) < MAX_RETRIES else "end"


def build_workflow():
    """Build and compile the LangGraph workflow."""
    graph = StateGraph(WorkflowState)
    
    graph.add_node("parse_extract", parse_extract_node)
    graph.add_node("generate", generate_node)
    graph.add_node("assemble", assemble_node)
    graph.add_node("validate", validate_node)
    graph.add_node("repair", repair_node)
    
    graph.set_entry_point("parse_extract")
    graph.add_edge("parse_extract", "generate")
    graph.add_edge("generate", "assemble")
    graph.add_edge("assemble", "validate")
    graph.add_conditional_edges("validate", should_repair, {"repair": "repair", "end": END})
    graph.add_edge("repair", "generate")
    
    return graph.compile()


workflow = build_workflow()
