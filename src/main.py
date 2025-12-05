"""CLI entry point for JSON re-contextualization."""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from .workflow import workflow, WorkflowState
from .config import OUTPUT_DIR


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO,
        stream=sys.stdout
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Re-contextualize JSON for a new scenario")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file")
    parser.add_argument("--scenario", "-s", required=True, help="Scenario index or string")
    parser.add_argument("--output-dir", "-o", default=OUTPUT_DIR)
    parser.add_argument("--list-scenarios", "-l", action="store_true")
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load input
    with open(args.input, "r") as f:
        input_json = json.load(f)
    
    # List scenarios mode
    if args.list_scenarios:
        scenarios = input_json.get("topicWizardData", {}).get("scenarioOptions", [])
        print(f"\nAvailable scenarios ({len(scenarios)}):\n")
        for i, s in enumerate(scenarios):
            print(f"  [{i}] {s[:80]}...")
        return
    
    # Parse scenario
    try:
        scenario = int(args.scenario)
    except ValueError:
        scenario = args.scenario
    
    # Run workflow
    logger.info("=" * 70)
    logger.info("STARTING RE-CONTEXTUALIZATION")
    logger.info("=" * 70)
    
    initial_state: WorkflowState = {
        "input_json": input_json,
        "selected_scenario": scenario,
        "old_scenario": "",
        "new_scenario": "",
        "locked_fields": {},
        "text_values": [],
        "output_json": None,
        "validation_result": None,
        "retry_count": 0,
        "failed_paths": [],
        "start_time": None,
        "llm_call_count": 0,
    }
    
    start = time.time()
    final_state = workflow.invoke(initial_state)
    elapsed = time.time() - start
    
    logger.info("=" * 70)
    logger.info(f"COMPLETED in {elapsed:.2f}s")
    logger.info("=" * 70)
    
    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, "output.json"), "w") as f:
        json.dump(final_state["output_json"], f, indent=2, ensure_ascii=False)
    
    with open(os.path.join(args.output_dir, "validation_report.json"), "w") as f:
        json.dump(final_state["validation_result"], f, indent=2)
    
    # Print summary
    result = final_state["validation_result"]
    print(f"\n✓ Output: {args.output_dir}/output.json")
    print(f"✓ Report: {args.output_dir}/validation_report.json")
    print(f"\nValidation: Schema={result['schemaMatch']}, Locked={result['lockedFieldsUntouched']}, Consistency={result['scenarioConsistency']}")
    print(f"Stats: {result['runtimeStats']['latency_ms']}ms, {result['runtimeStats']['numLLMCalls']} LLM calls, {result['runtimeStats']['numRetries']} retries")


if __name__ == "__main__":
    main()
