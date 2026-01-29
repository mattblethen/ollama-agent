import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_ollama import ollama_chat, ollama_tags
from agent_tools import (
    write_file,
    list_tree,
    read_text_snippet,
    list_project_files,
    local_verify_invariants,
    local_verify_grid_benchmark,
    ensure_dir,
)


STATE_FILE_NAME = ".agent_state.json"


def _now() -> float:
    return time.time()


def _safe_json_load(p: Path) -> Optional[dict]:
    try:
        if not p.exists():
            return None
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _safe_json_write(p: Path, obj: dict) -> None:
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _bounded_list(xs: List[str], n: int) -> List[str]:
    return xs[:n] if len(xs) > n else xs


def _summarize_changes(changes: List[dict], max_items: int = 20) -> Dict[str, Any]:
    """
    changes: output from agent_tools.write_file results
    Shape varies a bit, so keep it defensive.
    """
    out: Dict[str, Any] = {
        "count": 0,
        "files": [],
    }
    if not changes:
        return out
    files: List[str] = []
    for c in changes:
        if not isinstance(c, dict):
            continue
        p = c.get("path") or c.get("file") or c.get("relpath")
        if isinstance(p, str):
            files.append(p)
    files = sorted(set(files))
    out["count"] = len(files)
    out["files"] = _bounded_list(files, max_items)
    return out


def _derive_open_issues(verifier_report: Dict[str, Any], max_items: int = 12) -> List[str]:
    issues: List[str] = []
    if not isinstance(verifier_report, dict):
        return issues

    inv = verifier_report.get("invariants")
    if isinstance(inv, dict) and not inv.get("pass", True):
        for s in inv.get("issues") or []:
            if isinstance(s, str) and s.strip():
                issues.append(s.strip())

    plugs = verifier_report.get("plugins")
    if isinstance(plugs, dict) and not plugs.get("pass", True):
        for s in plugs.get("issues") or []:
            if isinstance(s, str) and s.strip():
                issues.append(s.strip())

    llm = verifier_report.get("llm")
    if isinstance(llm, dict) and not llm.get("pass", True):
        for s in llm.get("issues") or []:
            if isinstance(s, str) and s.strip():
                issues.append(s.strip())

    # Dedup + cap
    seen = set()
    cleaned: List[str] = []
    for s in issues:
        if s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
        if len(cleaned) >= max_items:
            break
    return cleaned


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

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

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
                chunk = text[start : i + 1]
                try:
                    obj = json.loads(chunk)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def _state_path(root: Path) -> Path:
    return root / STATE_FILE_NAME


def _load_state(root: Path) -> Optional[dict]:
    return _safe_json_load(_state_path(root))


def _write_state(root: Path, state: dict) -> None:
    _safe_json_write(_state_path(root), state)


def _build_context_pack(
    root: Path,
    task: str,
    plan_json: Optional[Dict[str, Any]],
    verifier_report: Optional[Dict[str, Any]],
    changes: Optional[List[dict]],
    run_dir: Optional[Path],
) -> Dict[str, Any]:
    """
    This is the *bounded* memory surface the planner should read.
    It prevents prompt bloat by keeping only the latest, summarized signal.
    """
    tree = list_tree(root)
    pack: Dict[str, Any] = {
        "version": 1,
        "updated_ts": _now(),
        "task": task,
        "run_dir": str(run_dir) if run_dir else "",
        "files_top": _bounded_list(tree, 250),
        "latest_plan": {
            "acceptance": (plan_json or {}).get("acceptance", []) if isinstance(plan_json, dict) else [],
            "plan": (plan_json or {}).get("plan", []) if isinstance(plan_json, dict) else [],
        },
        "latest_verifier": verifier_report if isinstance(verifier_report, dict) else {},
        "latest_changes": _summarize_changes(changes or [], max_items=30),
        "open_issues": _derive_open_issues(verifier_report or {}, max_items=12),
    }
    return pack


def _planner_prompt_fresh(task: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a planning assistant. Output ONLY JSON."},
        {
            "role": "user",
            "content": (
                "Create a short executable plan as JSON.\n\n"
                "Return JSON like:\n"
                "{\n"
                '  \"acceptance\": [\"...\"],\n'
                '  \"plan\": [{\"step\": 1, \"title\": \"...\", \"details\": \"...\", \"files_likely\": [\"...\"]}]\n'
                "}\n\n"
                "Constraints:\n"
                "- Keep the plan minimal.\n"
                "- Do NOT restate the task.\n"
                "- Aim for <= 8 steps.\n\n"
                f"TASK:\n{task}\n"
            ),
        },
    ]


def _planner_prompt_patch(task: str, context_pack: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Patch-mode planner: fast, bounded, and avoids re-planning the universe.
    """
    return [
        {"role": "system", "content": "You are a planning assistant. Output ONLY JSON."},
        {
            "role": "user",
            "content": (
                "You are continuing work in an existing project.\n"
                "Your job is to produce a MINIMAL patch plan to satisfy the task and resolve open issues.\n\n"
                "Return ONLY JSON with the same shape:\n"
                "{\n"
                '  \"acceptance\": [\"...\"],\n'
                '  \"plan\": [{\"step\": 1, \"title\": \"...\", \"details\": \"...\", \"files_likely\": [\"...\"]}]\n'
                "}\n\n"
                "Rules:\n"
                "- Do NOT rewrite the entire plan unless necessary.\n"
                "- Focus on the CURRENT open issues and the smallest set of file edits.\n"
                "- Prefer PATCH steps like 'edit X', 'adjust Y', 'add missing Z'.\n"
                "- If the project already meets acceptance, return plan=[] and acceptance unchanged.\n"
                "- Aim for <= 6 steps.\n"
                "- Do NOT restate the task.\n\n"
                f"TASK:\n{task}\n\n"
                "CONTEXT_PACK (latest snapshot, bounded):\n"
                f"{json.dumps(context_pack, ensure_ascii=False)}\n"
            ),
        },
    ]


def _coder_prompt(task: str, plan_json: Dict[str, Any], tree: List[str], focus: str) -> List[Dict[str, str]]:
    tree_text = "\n".join(tree[:300])
    return [
        {
            "role": "system",
            "content": (
                "You are a coding agent. You MUST respond with ONLY JSON.\n"
                "Your JSON must have this shape:\n"
                "{\n"
                '  \"writes\": [{\"path\":\"relative/path.ext\",\"content\":\"...\"}],\n'
                '  \"notes\": [\"...\"]\n'
                "}\n"
                "Rules:\n"
                "- Always include FULL FILE CONTENT in each write.\n"
                "- Paths must be relative. Do NOT write outside root.\n"
                "- If fixing something, rewrite the whole file(s).\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"TASK:\n{task}\n\n"
                f"PLAN_JSON:\n{json.dumps(plan_json, ensure_ascii=False)}\n\n"
                f"CURRENT_FILES:\n{tree_text}\n\n"
                f"FOCUS:\n{focus}\n\n"
                "Return only the JSON object."
            ),
        },
    ]


def _llm_verify_prompt(task: str, acceptance: List[str], tree: List[str], snippets: Dict[str, str]) -> List[Dict[str, str]]:
    acceptance_text = "\n".join(f"- {a}" for a in (acceptance or [])) or "(none provided)"
    return [
        {
            "role": "system",
            "content": (
                "You are a strict verifier. Decide pass/fail based on the acceptance criteria. Output ONLY JSON.\n"
                "Format:\n"
                "{\n"
                '  \"pass\": true/false,\n'
                '  \"issues\": [\"...\"],\n'
                '  \"suggestions\": [\"...\"],\n'
                '  \"notes\": \"optional\"\n'
                "}\n"
                "Rules:\n"
                "- Be specific and actionable.\n"
                "- If something is missing, say exactly what file/element/function is missing.\n"
                "- Do NOT invent files that do not exist.\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"TASK:\n{task}\n\n"
                f"ACCEPTANCE_CRITERIA:\n{acceptance_text}\n\n"
                f"FILES:\n" + "\n".join(tree[:350]) + "\n\n"
                "SNIPPETS:\n" + json.dumps(snippets, ensure_ascii=False) + "\n"
            ),
        },
    ]


def _chunk_task_via_llm(cfg: "RunCfg", task: str, run_dir: Path) -> List[str]:
    msgs = [
        {"role": "system", "content": "You split tasks into smaller steps. Output ONLY JSON."},
        {
            "role": "user",
            "content": (
                "Split the following task into 2-6 sequential subtasks that can be executed one-by-one.\n"
                "Return JSON: {\"subtasks\": [\"...\", \"...\"]}\n\n"
                f"TASK:\n{task}\n"
            ),
        },
    ]
    raw = ollama_chat(
        cfg.base_url,
        cfg.planner_model,
        msgs,
        temperature=0.2,
        timeout_s=min(cfg.timeout_s, 240),
        retries=cfg.retries,
    )
    _save_text(run_dir, "task_chunker_raw.txt", raw)
    obj = _extract_json_obj(raw) or {}
    subs = obj.get("subtasks") if isinstance(obj.get("subtasks"), list) else None
    if not subs:
        return [task[: cfg.chunk_max_chars]]
    cleaned: List[str] = []
    for s in subs:
        if not isinstance(s, str):
            continue
        s = s.strip()
        if not s:
            continue
        if len(s) > cfg.chunk_max_chars:
            s = s[: cfg.chunk_max_chars] + " (truncated)"
        cleaned.append(s)
    return cleaned[:6] if cleaned else [task[: cfg.chunk_max_chars]]


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
    verify_plugins: List[str]
    prompt_max_chars: int
    chunk_max_chars: int
    retries: int
    run_seed: str


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
    verify_plugins: List[str],
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
        verify_plugins=verify_plugins,
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

    # Load existing state (if any) to enable fast patch planning
    prior_state = _load_state(cfg.root)
    if prior_state:
        _log_event(run_dir, {"type": "note", "where": "state", "msg": f"Loaded {STATE_FILE_NAME} for patch planning"})

    effective_tasks = [cfg.task]
    if len(cfg.task) > cfg.prompt_max_chars:
        _log_event(run_dir, {"type": "note", "where": "prompt_guard", "msg": f"Task too large ({len(cfg.task)} chars) -> chunking"})
        effective_tasks = _chunk_task_via_llm(cfg, cfg.task, run_dir)
        _save_text(run_dir, "task_chunks.json", json.dumps({"subtasks": effective_tasks}, indent=2, ensure_ascii=False))

    for sub_i, subtask in enumerate(effective_tasks, start=1):
        focus_prefix = f"[subtask {sub_i}/{len(effective_tasks)}] "

        # Planner: if we have prior state, do PATCH-mode planning to avoid re-planning + reduce latency
        print(f"[agent] planner... {focus_prefix}")
        if prior_state:
            context_pack = dict(prior_state)
            # Refresh file list so patch planner sees current reality
            context_pack["files_top"] = _bounded_list(list_tree(cfg.root), 250)
            plan_msgs = _planner_prompt_patch(subtask, context_pack)
        else:
            plan_msgs = _planner_prompt_fresh(subtask)

        plan_raw = ollama_chat(
            cfg.base_url,
            cfg.planner_model,
            plan_msgs,
            temperature=0.2,
            timeout_s=cfg.timeout_s,
            retries=cfg.retries,
        )
        _save_text(run_dir, f"1_planner_raw_sub{sub_i}.txt", plan_raw)

        plan_json = _extract_json_obj(plan_raw) or {"acceptance": [], "plan": []}
        (run_dir / f"1_plan_sub{sub_i}.json").write_text(json.dumps(plan_json, indent=2, ensure_ascii=False), encoding="utf-8")
        _log_event(run_dir, {"type": "plan", "sub": sub_i, "plan": plan_json.get("plan", []), "acceptance": plan_json.get("acceptance", [])})

        # Immediately write/update state after planning (so next run starts bounded)
        state_after_plan = _build_context_pack(
            cfg.root,
            cfg.task,
            plan_json=plan_json,
            verifier_report=None,
            changes=None,
            run_dir=run_dir,
        )
        _write_state(cfg.root, state_after_plan)

        last_progress = _now()
        last_verifier_report: Dict[str, Any] = {}
        last_changes: List[dict] = []

        for it in range(1, cfg.max_iters + 1):
            _log_event(run_dir, {"type": "iter_start", "iter": it, "sub": sub_i})
            print(f"[agent] Iteration {it}/{cfg.max_iters} {focus_prefix}")

            if (_now() - last_progress) > cfg.stall_s:
                _log_event(run_dir, {"type": "stall_abort", "iter": it, "sub": sub_i, "stall_s": cfg.stall_s})
                raise RuntimeError(f"Hard stall detected: no progress for {cfg.stall_s}s")

            tree = list_tree(cfg.root)

            # --- Verification stack (project-agnostic by default) ---
            ok_inv, issues_inv = local_verify_invariants(cfg.root)
            _log_event(run_dir, {"type": "verify_local", "iter": it, "sub": sub_i, "name": "invariants", "pass": ok_inv, "issues": issues_inv})

            plugin_issues: List[str] = []
            plugin_pass = True
            for plug in (cfg.verify_plugins or []):
                if plug == "grid_benchmark":
                    okp, isp = local_verify_grid_benchmark(cfg.root)
                else:
                    okp, isp = False, [f"Unknown verify plugin: {plug}"]
                plugin_pass = plugin_pass and okp
                plugin_issues.extend([f"[{plug}] {x}" for x in isp])
                _log_event(run_dir, {"type": "verify_local", "iter": it, "sub": sub_i, "name": plug, "pass": okp, "issues": isp})

            # LLM verifier is the primary acceptance gate
            vobj: Dict[str, Any] = {"pass": False, "issues": [], "suggestions": []}
            ok_llm = False
            llm_issues: List[str] = []
            llm_suggestions: List[str] = []

            if cfg.llm_verifier:
                proj_files = list_project_files(cfg.root, max_files=80)
                prefer = ["README.md", "index.html", "main.py", "app.py", "package.json", "style.css", "script.js"]
                chosen: List[str] = []
                for p in prefer:
                    if p in proj_files and p not in chosen:
                        chosen.append(p)
                for p in proj_files:
                    if p in chosen:
                        continue
                    if Path(p).suffix.lower() in {".md", ".txt", ".py", ".js", ".ts", ".html", ".css", ".json"}:
                        chosen.append(p)
                    if len(chosen) >= 8:
                        break

                snippets = {p: read_text_snippet(cfg.root, p, 7000) for p in chosen}
                _log_event(run_dir, {"type": "note", "where": "verifier_request", "iter": it, "sub": sub_i, "model": cfg.verifier_model, "files": chosen})
                vraw = ollama_chat(
                    cfg.base_url,
                    cfg.verifier_model,
                    _llm_verify_prompt(subtask, list(plan_json.get("acceptance", []) or []), tree, snippets),
                    temperature=0.1,
                    timeout_s=cfg.timeout_s,
                    retries=cfg.retries,
                )
                _save_text(run_dir, f"4_verifier_raw_sub{sub_i}_iter{it}.txt", vraw)
                vobj = _extract_json_obj(vraw) or {"pass": False, "issues": ["Verifier returned non-JSON"], "suggestions": []}
                (run_dir / f"4_verifier_sub{sub_i}_iter{it}.json").write_text(json.dumps(vobj, indent=2, ensure_ascii=False), encoding="utf-8")
                ok_llm = bool(vobj.get("pass"))
                llm_issues = [x for x in (vobj.get("issues") or []) if isinstance(x, str)]
                llm_suggestions = [x for x in (vobj.get("suggestions") or []) if isinstance(x, str)]
                _log_event(run_dir, {"type": "verify_llm", "iter": it, "sub": sub_i, "pass": ok_llm, "issues": llm_issues, "suggestions": llm_suggestions})

            if ok_inv and plugin_pass and (ok_llm if cfg.llm_verifier else True):
                verifier_report = {
                    "invariants": {"pass": ok_inv, "issues": issues_inv},
                    "plugins": {"pass": plugin_pass, "issues": plugin_issues},
                    "llm": vobj if cfg.llm_verifier else {"pass": True, "issues": [], "suggestions": []},
                }
                _log_event(
                    run_dir,
                    {
                        "type": "summary",
                        "verified_pass": True,
                        "sub": sub_i,
                        "verifier_report": verifier_report,
                    },
                )
                # Update state on success too
                final_state = _build_context_pack(
                    cfg.root,
                    cfg.task,
                    plan_json=plan_json,
                    verifier_report=verifier_report,
                    changes=last_changes,
                    run_dir=run_dir,
                )
                _write_state(cfg.root, final_state)

                print("[agent] done. verified_pass=True")
                break

            last_verifier_report = {
                "invariants": {"pass": ok_inv, "issues": issues_inv},
                "plugins": {"pass": plugin_pass, "issues": plugin_issues, "enabled": list(cfg.verify_plugins or [])},
                "llm": vobj if cfg.llm_verifier else {"pass": True, "issues": [], "suggestions": []},
            }

            # Update state snapshot every iteration (bounded)
            iter_state = _build_context_pack(
                cfg.root,
                cfg.task,
                plan_json=plan_json,
                verifier_report=last_verifier_report,
                changes=last_changes,
                run_dir=run_dir,
            )
            _write_state(cfg.root, iter_state)

            focus_lines: List[str] = []
            if not ok_inv:
                focus_lines.extend(issues_inv)
            if not plugin_pass:
                focus_lines.extend(plugin_issues)
            if cfg.llm_verifier and not ok_llm:
                focus_lines.extend(llm_issues)
                if llm_suggestions:
                    focus_lines.append("--- Suggestions (hints) ---")
                    focus_lines.extend(llm_suggestions[:5])

            if not focus_lines:
                focus_lines.append("Verifier did not provide actionable issues. Make a small, safe improvement toward the acceptance criteria.")

            focus = "Fix the following issues (in order). Do NOT introduce new files/features unless required.\n- " + "\n- ".join(focus_lines[:10])
            msgs = _coder_prompt(subtask, plan_json, tree, focus)

            _log_event(run_dir, {"type": "note", "where": "coder_request", "iter": it, "sub": sub_i, "model": cfg.coder_model, "timeout_s": cfg.timeout_s})
            coder_raw = ollama_chat(
                cfg.base_url,
                cfg.coder_model,
                msgs,
                temperature=0.2,
                timeout_s=cfg.timeout_s,
                retries=cfg.retries,
            )
            _save_text(run_dir, f"2_coder_raw_sub{sub_i}_iter{it}.txt", coder_raw)
            _log_event(run_dir, {"type": "note", "where": "coder_response", "iter": it, "sub": sub_i})

            obj = _extract_json_obj(coder_raw) or {}
            writes = obj.get("writes") if isinstance(obj.get("writes"), list) else []

            if not writes:
                _log_event(
                    run_dir,
                    {
                        "type": "coder_no_writes",
                        "iter": it,
                        "sub": sub_i,
                        "msg": "Coder returned zero writes. Next iteration should explicitly write required files.",
                    },
                )

            changes: List[dict] = []
            for w in writes:
                if not isinstance(w, dict):
                    continue
                path = w.get("path")
                content = w.get("content")
                if not isinstance(path, str) or not isinstance(content, str):
                    continue
                changes.append(write_file(cfg.root, path, content))

            last_changes = changes

            _log_event(run_dir, {"type": "tool_results", "iter": it, "sub": sub_i, "changes": changes})
            (run_dir / f"3_tool_results_sub{sub_i}_iter{it}.json").write_text(
                json.dumps({"changes": changes}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Update state after writes too (helps next run immediately)
            post_write_state = _build_context_pack(
                cfg.root,
                cfg.task,
                plan_json=plan_json,
                verifier_report=last_verifier_report,
                changes=last_changes,
                run_dir=run_dir,
            )
            _write_state(cfg.root, post_write_state)

            if changes:
                last_progress = _now()

        else:
            _log_event(
                run_dir,
                {
                    "type": "summary",
                    "verified_pass": False,
                    "sub": sub_i,
                    "verifier_report": {
                        "pass": False,
                        "reasons": ["max iters"],
                        "last": last_verifier_report,
                    },
                },
            )

            # Write state even on failure (for patch planning next run)
            fail_state = _build_context_pack(
                cfg.root,
                cfg.task,
                plan_json=plan_json,
                verifier_report=last_verifier_report,
                changes=last_changes,
                run_dir=run_dir,
            )
            _write_state(cfg.root, fail_state)

            print("[agent] done. verified_pass=False (max iters)")
            return 1

    return 0
