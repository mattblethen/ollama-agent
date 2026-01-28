#!/usr/bin/env python3
import argparse
from agent_core import run_agent

def main():
    ap = argparse.ArgumentParser(description="Ollama Agent (modular v3)")
    ap.add_argument("--root", required=True, help="Project root folder")
    ap.add_argument("--task", required=True, help="Task prompt")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434", help="Ollama base URL")
    ap.add_argument("--planner", default="qwen3:30b", help="Planner model")
    ap.add_argument("--coder", default="qwen3-coder:30b", help="Coder model")
    ap.add_argument("--verifier", default="qwen3:30b", help="Verifier model (optional)")
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=180, help="Per-call timeout seconds (planner/coder/verifier)")
    ap.add_argument("--stall", type=int, default=240, help="Hard stall detector seconds (no new events)")
    ap.add_argument("--no-llm-verifier", action="store_true", help="Disable LLM verifier; use local checks only")
    ap.add_argument("--prompt-max-chars", type=int, default=16000, help="Prompt size guard threshold (chars)")
    ap.add_argument("--chunk-max-chars", type=int, default=9000, help="Max chars per chunked subtask")
    ap.add_argument("--retries", type=int, default=2, help="Retries for Ollama calls")
    ap.add_argument("--seed", default="", help="Optional run seed label (for logging only)")
    args = ap.parse_args()

    return run_agent(
        root=args.root,
        task=args.task,
        base_url=args.ollama,
        planner_model=args.planner,
        coder_model=args.coder,
        verifier_model=args.verifier,
        max_iters=args.max_iters,
        timeout_s=args.timeout,
        stall_s=args.stall,
        llm_verifier=not args.no_llm_verifier,
        prompt_max_chars=args.prompt_max_chars,
        chunk_max_chars=args.chunk_max_chars,
        retries=args.retries,
        run_seed=args.seed,
    )

if __name__ == "__main__":
    raise SystemExit(main())
