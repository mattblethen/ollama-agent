#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from datetime import datetime

def load_events(events_path: Path):
    events = []
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events

def summarize(events):
    summary = {
        "root": None,
        "task": None,
        "ollama_base_url": None,
        "models": {},
        "iterations": 0,
        "verified_pass": None,
        "verifier_report": None,
        "files_written": [],
        "files_deleted": [],
        "moves": [],
        "mkdirs": [],
        "git_commits": [],
        "notes": [],
    }

    last_plan = None
    last_verifier = None
    last_summary_evt = None

    for e in events:
        et = e.get("type")

        if et == "start":
            summary["root"] = e.get("root")
            summary["task"] = e.get("task")
            summary["ollama_base_url"] = e.get("ollama_base_url")

        elif et == "models":
            summary["models"] = {
                "planner": e.get("planner"),
                "coder": e.get("coder"),
                "verifier": e.get("verifier"),
            }

        elif et == "iter_start":
            try:
                summary["iterations"] = max(summary["iterations"], int(e.get("iter", 0)))
            except Exception:
                pass

        elif et == "plan":
            last_plan = {"plan": e.get("plan", []), "acceptance": e.get("acceptance", [])}

        elif et == "tool_results":
            changes = e.get("changes", []) or []
            for c in changes:
                op = c.get("op")
                if op == "write_file":
                    summary["files_written"].append({
                        "path": c.get("path"),
                        "existed": c.get("existed"),
                        "old_hash": c.get("old_hash"),
                        "new_hash": c.get("new_hash"),
                    })
                elif op == "delete_path":
                    summary["files_deleted"].append({"path": c.get("path"), "existed": c.get("existed")})
                elif op == "move_path":
                    summary["moves"].append({"src": c.get("src"), "dst": c.get("dst")})
                elif op == "mkdir":
                    summary["mkdirs"].append({"path": c.get("path")})
                elif op == "git_commit":
                    summary["git_commits"].append({"message": c.get("message"), "ok": c.get("ok")})

        elif et == "verifier":
            last_verifier = {
                "pass": e.get("pass"),
                "issues": e.get("issues", []),
                "suggestions": e.get("suggestions", []),
                "next_action_hint": e.get("next_action_hint"),
                "notes": e.get("notes"),
            }

        elif et == "summary":
            last_summary_evt = e
            summary["verified_pass"] = e.get("verified_pass")
            summary["verifier_report"] = e.get("verifier_report")

        elif et in ("note", "error", "stall_abort"):
            summary["notes"].append(e)

    return summary, last_plan, last_verifier, last_summary_evt

def write_md(out_path: Path, summary, plan, verifier, run_dir: Path):
    lines = []
    lines.append(f"# Agent Run Report\n")
    lines.append(f"- Run dir: `{run_dir}`")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Root: `{summary.get('root')}`")
    lines.append(f"- Ollama: `{summary.get('ollama_base_url')}`")
    lines.append(f"- Models: {json.dumps(summary.get('models', {}), ensure_ascii=False)}")
    lines.append(f"- Iterations: {summary.get('iterations')}")
    lines.append(f"- Verified pass: **{summary.get('verified_pass')}**\n")

    if summary.get("task"):
        lines.append("## Task")
        lines.append(summary["task"])
        lines.append("")

    if plan:
        lines.append("## Plan & Acceptance")
        lines.append("### Acceptance")
        for a in plan.get("acceptance", []):
            lines.append(f"- {a}")
        lines.append("\n### Plan")
        for s in plan.get("plan", []):
            lines.append(f"- Step {s.get('step')}: **{s.get('title')}** — {s.get('details')}")
        lines.append("")

    if verifier:
        lines.append("## Final Verifier Verdict (event stream)")
        lines.append(f"- pass: **{verifier.get('pass')}**")
        issues = verifier.get("issues") or []
        if issues:
            lines.append("### Issues")
            for i in issues:
                lines.append(f"- {i}")
        lines.append("")

    lines.append("## File Operations")
    fw = summary.get("files_written", [])
    lines.append(f"- Writes: {len(fw)}")
    for w in fw[-50:]:
        lines.append(f"  - `{w.get('path')}` existed={w.get('existed')} new_hash={w.get('new_hash')}")
    lines.append("")

    if summary.get("notes"):
        lines.append("## Notes / Errors / Stalls")
        for n in summary["notes"][-60:]:
            lines.append(f"- {json.dumps(n, ensure_ascii=False)}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")

def main():
    if len(sys.argv) < 2:
        print("Usage: compile_run.py /path/to/.agent_logs/<runid>", file=sys.stderr)
        return 2

    run_dir = Path(sys.argv[1]).expanduser().resolve()
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        print(f"events.jsonl not found in {run_dir}", file=sys.stderr)
        return 3

    events = load_events(events_path)
    summary, plan, verifier, _ = summarize(events)

    (run_dir / "MASTER_RUN_REPORT.json").write_text(
        json.dumps({"summary": summary, "plan": plan, "final_verifier": verifier}, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    write_md(run_dir / "MASTER_RUN_REPORT.md", summary, plan, verifier, run_dir)

    print(f"Wrote:\n- {run_dir / 'MASTER_RUN_REPORT.md'}\n- {run_dir / 'MASTER_RUN_REPORT.json'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
