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


class WorkflowState(TypedDict):
    input_json: Dict[str, Any]
    selected_scenario: Union[str, int]
    old_scenario: str
    new_scenario: str
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


def get_old_brand_names(old_scenario: str, new_scenario: str) -> List[str]:
    """Extract brand/company names from OLD scenario that shouldn't appear in output.
    
    Only looks for brand-like patterns, not generic words.
    """
    old_brands = set(extract_scenario_brands(old_scenario))
    new_brands = set(extract_scenario_brands(new_scenario))
    
    # Return lowercase versions for case-insensitive matching
    return [b.lower() for b in (old_brands - new_brands) if len(b) > 3]


def get_compressed_scenario_info(old_scenario: str, new_scenario: str) -> tuple[str, str]:
    """Extract compressed brand info for prompts instead of full scenarios.
    
    Returns (old_brands_str, new_brands_str) for use in prompts.
    """
    old_brands = extract_scenario_brands(old_scenario)
    new_brands = extract_scenario_brands(new_scenario)
    
    old_str = ", ".join(old_brands) if old_brands else old_scenario[:100]
    new_str = ", ".join(new_brands) if new_brands else new_scenario[:100]
    
    return old_str, new_str


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
    """Extract ALL text values and save locked fields."""
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
    
    return {
        "old_scenario": old_scenario,
        "new_scenario": new_scenario,
        "locked_fields": locked_fields,
        "text_values": text_values,
        "start_time": start_time,
        "llm_call_count": 0,
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
    old_brands: str,
    new_brands: str,
    batch_num: int,
    is_large_text: bool = False
) -> List[str]:
    """Rewrite a batch of texts using LLM.
    
    Uses compressed brand info for small batches, full scenarios for large texts.
    """
    logger.info(f"      [Batch {batch_num}] Starting ({len(batch)} texts, large={is_large_text})...")
    start = time.time()
    
    # Format texts with numbers - FULL TEXT, NO SLICING
    text_list = "\n".join([f"[{i+1}] {t['original']}" for i, t in enumerate(batch)])
    
    # Always use full scenarios - compressed brand names caused quality issues
    # (LLM confused "Buy One Get One" promotion as a company name)
    scenario_info = f"OLD SCENARIO: {old_scenario}\n\nNEW SCENARIO: {new_scenario}"
    
    # Compact prompt to reduce tokens
    system_prompt = f"""Rewrite texts replacing old scenario references with new ones.

RULES:
1. Replace ALL old references (brands, companies, emails, URLs, names)
2. Keep structure, tone, length, formatting (HTML/markdown intact)
3. Generic content stays unchanged
4. Return each text as [1], [2], etc. on its own line

{scenario_info}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"TEXTS:\n{text_list}\n\nRewritten:"),
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
    """Rewrite text values in parallel batches."""
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
    
    # Get compressed scenario info for prompts
    old_brands, new_brands = get_compressed_scenario_info(state["old_scenario"], state["new_scenario"])
    logger.info(f"  Old brands: {old_brands[:80]}...")
    logger.info(f"  New brands: {new_brands[:80]}...")
    
    llm = ChatOpenAI(model=GENERATION_MODEL, temperature=MODEL_TEMPERATURE, api_key=OPENAI_API_KEY)
    
    async def run_all():
        tasks = []
        for i, b in enumerate(batches):
            # Check if this is a section from a split text (needs full context)
            is_section = len(b) == 1 and "_parent_path" in b[0]
            is_large = len(b) == 1 and len(b[0]["original"]) > LARGE_TEXT_THRESHOLD
            tasks.append(rewrite_batch(
                llm, b, 
                state["old_scenario"], state["new_scenario"],
                old_brands, new_brands,
                i+1, is_large or is_section  # Sections get full scenario context
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
    """Validate schema, locked fields, and check for old scenario leakage."""
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
    
    # 3. Leakage check - find old BRAND NAMES in output (not generic words)
    old_brands = get_old_brand_names(state["old_scenario"], state["new_scenario"])
    
    failed_paths = []
    for t in state.get("text_values", []):
        if "scenarioOptions" in t["path"]:
            continue  # Skip locked field
        if t.get("rewritten"):
            text_lower = t["rewritten"].lower()
            for brand in old_brands:
                if brand in text_lower:
                    failed_paths.append(t["path"])
                    break
    
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
