import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_ollama import ollama_chat, ollama_tags
from agent_tools import write_file, list_tree, read_text_snippet, local_verify_web_grid_project, ensure_dir

def _now() -> float:
    return time.time()

@dataclass
class RunCfg:
    root: Path
    task: str
    base_url: str
    planner_model: str
    coder_model: str
    verifier_model: str
    max_iters: int
    timeout_s: int
    stall_s: int
    llm_verifier: bool
    prompt_max_chars: int
    chunk_max_chars: int
    retries: int
    run_seed: str

def _log_event(run_dir: Path, evt: Dict[str, Any]) -> None:
    evt = dict(evt)
    evt["ts"] = evt.get("ts", _now())
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(evt, ensure_ascii=False) + "\n")

def _save_text(run_dir: Path, name: str, content: str) -> None:
    (run_dir / name).write_text(content, encoding="utf-8")

def _extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract first top-level JSON object from model output.
    Accepts raw JSON or text containing JSON.
    """
    text = text.strip()
    if not text:
        return None
    # fast path
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # find first '{' ... matching '}'
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start:i+1]
                try:
                    obj = json.loads(chunk)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None

def _planner_prompt(task: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a planning assistant. Output ONLY JSON."},
        {"role": "user", "content":
            "Create a short executable plan as JSON.\n\n"
            "Return JSON like:\n"
            "{\n"
            '  "acceptance": ["..."],\n'
            '  "plan": [{"step": 1, "title": "...", "details": "...", "files_likely": ["..."]}]\n'
            "}\n\n"
            f"TASK:\n{task}\n"
        }
    ]

def _coder_prompt(task: str, plan_json: Dict[str, Any], tree: List[str], focus: str) -> List[Dict[str, str]]:
    tree_text = "\n".join(tree[:300])
    return [
        {"role": "system", "content":
            "You are a coding agent. You MUST respond with ONLY JSON.\n"
            "Your JSON must have this shape:\n"
            "{\n"
            '  "writes": [{"path":"relative/path.ext","content":"..."}],\n'
            '  "notes": ["..."]\n'
            "}\n"
            "Rules:\n"
            "- Always include FULL FILE CONTENT in each write.\n"
            "- Paths must be relative. Do NOT write outside root.\n"
            "- If fixing something, rewrite the whole file(s).\n"
        },
        {"role": "user", "content":
            f"TASK:\n{task}\n\n"
            f"PLAN_JSON:\n{json.dumps(plan_json, ensure_ascii=False)}\n\n"
            f"CURRENT_FILES:\n{tree_text}\n\n"
            f"FOCUS:\n{focus}\n\n"
            "Return only the JSON object."
        }
    ]

def _llm_verify_prompt(task: str, tree: List[str], snippets: Dict[str, str]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content":
            "You are a strict verifier. Decide pass/fail. Output ONLY JSON.\n"
            "Format:\n"
            "{\n"
            '  "pass": true/false,\n'
            '  "reasons": ["..."]\n'
            "}\n"
        },
        {"role": "user", "content":
            f"TASK:\n{task}\n\n"
            f"FILES:\n" + "\n".join(tree[:250]) + "\n\n"
            "SNIPPETS:\n" + json.dumps(snippets, ensure_ascii=False) + "\n"
        }
    ]

def _chunk_task_via_llm(cfg: RunCfg, task: str, run_dir: Path) -> List[str]:
    """
    Ask planner model to chunk huge tasks into smaller sequential subtasks.
    """
    msgs = [
        {"role": "system", "content": "You split tasks into smaller steps. Output ONLY JSON."},
        {"role": "user", "content":
            "Split the following task into 2-6 sequential subtasks that can be executed one-by-one.\n"
            "Return JSON: {\"subtasks\": [\"...\", \"...\"]}\n\n"
            f"TASK:\n{task}\n"
        }
    ]
    raw = ollama_chat(cfg.base_url, cfg.planner_model, msgs, temperature=0.2,
                      timeout_s=min(cfg.timeout_s, 240), retries=cfg.retries)
    _save_text(run_dir, "task_chunker_raw.txt", raw)
    obj = _extract_json_obj(raw) or {}
    subs = obj.get("subtasks") if isinstance(obj.get("subtasks"), list) else None
    if not subs:
        # fallback naive split
        return [task[:cfg.chunk_max_chars]]
    cleaned = []
    for s in subs:
        if not isinstance(s, str):
            continue
        s = s.strip()
        if not s:
            continue
        if len(s) > cfg.chunk_max_chars:
            s = s[:cfg.chunk_max_chars] + " (truncated)"
        cleaned.append(s)
    return cleaned[:6] if cleaned else [task[:cfg.chunk_max_chars]]

def run_agent(
    root: str,
    task: str,
    base_url: str,
    planner_model: str,
    coder_model: str,
    verifier_model: str,
    max_iters: int,
    timeout_s: int,
    stall_s: int,
    llm_verifier: bool,
    prompt_max_chars: int,
    chunk_max_chars: int,
    retries: int,
    run_seed: str = "",
) -> int:
    cfg = RunCfg(
        root=Path(root).expanduser().resolve(),
        task=task,
        base_url=base_url,
        planner_model=planner_model,
        coder_model=coder_model,
        verifier_model=verifier_model,
        max_iters=max_iters,
        timeout_s=timeout_s,
        stall_s=stall_s,
        llm_verifier=llm_verifier,
        prompt_max_chars=prompt_max_chars,
        chunk_max_chars=chunk_max_chars,
        retries=retries,
        run_seed=run_seed,
    )

    ensure_dir(cfg.root)
    run_dir = cfg.root / ".agent_logs" / time.strftime("%Y%m%d-%H%M%S")
    ensure_dir(run_dir)

    print(f"[agent] run_dir = {run_dir}")
    print(f"[agent] root    = {cfg.root}")
    print(f"[agent] ollama  = {cfg.base_url}")
    print(f"[agent] models: planner={cfg.planner_model} coder={cfg.coder_model} verifier={cfg.verifier_model}")

    _log_event(run_dir, {"type": "start", "root": str(cfg.root), "task": cfg.task, "ollama_base_url": cfg.base_url})
    _log_event(run_dir, {"type": "models", "planner": cfg.planner_model, "coder": cfg.coder_model, "verifier": cfg.verifier_model})

    # quick connectivity sanity
    try:
        _ = ollama_tags(cfg.base_url, timeout_s=15)
    except Exception as e:
        _log_event(run_dir, {"type": "error", "where": "ollama_tags", "error": str(e)})
        raise

    # prompt size guard + chunking
    effective_tasks = [cfg.task]
    if len(cfg.task) > cfg.prompt_max_chars:
        _log_event(run_dir, {"type": "note", "where": "prompt_guard", "msg": f"Task too large ({len(cfg.task)} chars) -> chunking"})
        effective_tasks = _chunk_task_via_llm(cfg, cfg.task, run_dir)
        _save_text(run_dir, "task_chunks.json", json.dumps({"subtasks": effective_tasks}, indent=2, ensure_ascii=False))

    # Execute subtasks sequentially in one run
    for sub_i, subtask in enumerate(effective_tasks, start=1):
        focus_prefix = f"[subtask {sub_i}/{len(effective_tasks)}] "

        print(f"[agent] planner... {focus_prefix}")
        plan_raw = ollama_chat(cfg.base_url, cfg.planner_model, _planner_prompt(subtask),
                               temperature=0.2, timeout_s=cfg.timeout_s, retries=cfg.retries)
        _save_text(run_dir, f"1_planner_raw_sub{sub_i}.txt", plan_raw)

        plan_json = _extract_json_obj(plan_raw) or {"acceptance": [], "plan": []}
        (run_dir / f"1_plan_sub{sub_i}.json").write_text(json.dumps(plan_json, indent=2, ensure_ascii=False), encoding="utf-8")
        _log_event(run_dir, {"type": "plan", "sub": sub_i, "plan": plan_json.get("plan", []), "acceptance": plan_json.get("acceptance", [])})

        last_progress = _now()

        for it in range(1, cfg.max_iters + 1):
            _log_event(run_dir, {"type": "iter_start", "iter": it, "sub": sub_i})
            print(f"[agent] Iteration {it}/{cfg.max_iters} {focus_prefix}")

            # Hard stall detector (no new writes/events)
            if (_now() - last_progress) > cfg.stall_s:
                _log_event(run_dir, {"type": "stall_abort", "iter": it, "sub": sub_i, "stall_s": cfg.stall_s})
                raise RuntimeError(f"Hard stall detected: no progress for {cfg.stall_s}s")

            tree = list_tree(cfg.root)

            # Local deterministic verifier first
            ok_local, issues_local = local_verify_web_grid_project(cfg.root)
            if ok_local:
                # optional LLM verifier
                if cfg.llm_verifier:
                    snippets = {
                        "index.html": read_text_snippet(cfg.root, "index.html", 6000),
                        "style.css": read_text_snippet(cfg.root, "style.css", 6000),
                        "script.js": read_text_snippet(cfg.root, "script.js", 9000),
                    }
                    _log_event(run_dir, {"type": "note", "where": "verifier_request", "iter": it, "sub": sub_i, "model": cfg.verifier_model})
                    vraw = ollama_chat(cfg.base_url, cfg.verifier_model, _llm_verify_prompt(subtask, tree, snippets),
                                      temperature=0.1, timeout_s=cfg.timeout_s, retries=cfg.retries)
                    _save_text(run_dir, f"4_verifier_raw_sub{sub_i}_iter{it}.txt", vraw)
                    vobj = _extract_json_obj(vraw) or {"pass": False, "reasons": ["Verifier returned non-JSON"]}
                    (run_dir / f"4_verifier_sub{sub_i}_iter{it}.json").write_text(json.dumps(vobj, indent=2, ensure_ascii=False), encoding="utf-8")
                    _log_event(run_dir, {"type": "verifier", "pass": bool(vobj.get("pass")), "issues": vobj.get("reasons", [])})

                    if bool(vobj.get("pass")):
                        _log_event(run_dir, {"type": "summary", "verified_pass": True, "sub": sub_i, "verifier_report": vobj})
                        print("[agent] done. verified_pass=True")
                        break
                else:
                    _log_event(run_dir, {"type": "summary", "verified_pass": True, "sub": sub_i, "verifier_report": {"pass": True, "reasons": []}})
                    print("[agent] done. verified_pass=True (local)")
                    break

            # If local verifier fails, instruct coder explicitly to fix issues.
            focus = "Fix the following issues:\n- " + "\n- ".join(issues_local[:12])
            msgs = _coder_prompt(subtask, plan_json, tree, focus)

            _log_event(run_dir, {"type": "note", "where": "coder_request", "iter": it, "sub": sub_i, "model": cfg.coder_model, "timeout_s": cfg.timeout_s})
            coder_raw = ollama_chat(cfg.base_url, cfg.coder_model, msgs,
                                    temperature=0.2, timeout_s=cfg.timeout_s, retries=cfg.retries)
            _save_text(run_dir, f"2_coder_raw_sub{sub_i}_iter{it}.txt", coder_raw)
            _log_event(run_dir, {"type": "note", "where": "coder_response", "iter": it, "sub": sub_i})

            obj = _extract_json_obj(coder_raw) or {}
            writes = obj.get("writes") if isinstance(obj.get("writes"), list) else []

            changes = []
            for w in writes:
                if not isinstance(w, dict):
                    continue
                path = w.get("path")
                content = w.get("content")
                if not isinstance(path, str) or not isinstance(content, str):
                    continue
                changes.append(write_file(cfg.root, path, content))

            _log_event(run_dir, {"type": "tool_results", "iter": it, "sub": sub_i, "changes": changes})
            (run_dir / f"3_tool_results_sub{sub_i}_iter{it}.json").write_text(json.dumps({"changes": changes}, indent=2, ensure_ascii=False), encoding="utf-8")

            if changes:
                last_progress = _now()

        else:
            # max iters exceeded for subtask
            _log_event(run_dir, {"type": "summary", "verified_pass": False, "sub": sub_i, "verifier_report": {"pass": False, "reasons": ["max iters"]}})
            print("[agent] done. verified_pass=False (max iters)")
            return 1

    return 0
