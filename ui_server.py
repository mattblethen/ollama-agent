#!/usr/bin/env python3
"""
Minimal Web UI server for ollama-agent
- Serves UI at /
- Lists local Ollama models
- Starts agent runs (subprocess calling agent.py)
- Streams stdout + events.jsonl lines via SSE
- Supports Continue using per-root saved config

No external deps. Python 3.9+ recommended.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_DIR = Path(__file__).resolve().parent
UI_DIR = REPO_DIR / "ui"
STATE_FILE_NAME = ".agent_ui_state.json"

RUNS_LOCK = threading.Lock()
RUNS: Dict[str, "RunState"] = {}  # run_id -> state


def now_ms() -> int:
    return int(time.time() * 1000)


def json_read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def json_write(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def http_get_json(url: str, timeout_s: int = 10) -> Any:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8", errors="replace"))


def list_ollama_models(base_url: str, timeout_s: int = 10) -> Tuple[bool, Any]:
    # Ollama: GET /api/tags -> {"models":[{"name":"...","model":"...","modified_at":"..."}...]}
    try:
        data = http_get_json(base_url.rstrip("/") + "/api/tags", timeout_s=timeout_s)
        models = []
        for m in data.get("models", []):
            name = m.get("name") or m.get("model")
            if isinstance(name, str):
                models.append(name)
        models = sorted(set(models))
        return True, {"models": models, "raw": data}
    except Exception as e:
        return False, {"error": str(e)}


def find_latest_run_dir(root: Path) -> Optional[Path]:
    logs = root / ".agent_logs"
    if not logs.exists():
        return None
    dirs = [p for p in logs.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0]


def tail_file_lines(path: Path, start_pos: int) -> Tuple[int, list[str]]:
    """Return (new_pos, new_lines) reading from start_pos to EOF."""
    if not path.exists():
        return start_pos, []
    with path.open("rb") as f:
        f.seek(start_pos)
        data = f.read()
        new_pos = f.tell()
    if not data:
        return new_pos, []
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return new_pos, lines


@dataclass
class RunConfig:
    ollama: str
    root: str
    task: str
    planner: str
    coder: str
    verifier: str
    max_iters: int = 12
    timeout: int = 900

    # Optional knobs (pass-through if your agent supports them; ignored otherwise)
    stall: Optional[int] = None
    no_llm_verifier: Optional[bool] = None
    verify_plugins: Optional[list[str]] = None


@dataclass
class RunState:
    run_id: str
    cfg: RunConfig
    started_ms: int
    finished_ms: Optional[int] = None
    exit_code: Optional[int] = None
    run_dir: Optional[str] = None

    # streaming
    q: "queue.Queue[dict]" = queue.Queue()
    stop_evt: threading.Event = threading.Event()

    # internal
    _proc: Optional[subprocess.Popen] = None
    _events_pos: int = 0


def save_root_state(root: Path, cfg: RunConfig) -> None:
    path = root / STATE_FILE_NAME
    json_write(path, {"last_cfg": asdict(cfg), "saved_ms": now_ms()})


def load_root_state(root: Path) -> Optional[RunConfig]:
    path = root / STATE_FILE_NAME
    data = json_read(path, None)
    if not isinstance(data, dict):
        return None
    last = data.get("last_cfg")
    if not isinstance(last, dict):
        return None
    try:
        return RunConfig(**last)
    except Exception:
        return None


def push(rs: RunState, typ: str, payload: Any) -> None:
    rs.q.put({"t": typ, "d": payload, "ts": now_ms()})


def run_agent_subprocess(rs: RunState) -> None:
    """Launch agent.py as a subprocess, stream stdout, and tail events.jsonl."""
    root = Path(rs.cfg.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    save_root_state(root, rs.cfg)

    # Build command (use only flags your current agent.py supports; unknown flags will error)
    cmd = [
        "python3",
        str(REPO_DIR / "agent.py"),
        "--ollama",
        rs.cfg.ollama,
        "--root",
        rs.cfg.root,
        "--task",
        rs.cfg.task,
        "--planner",
        rs.cfg.planner,
        "--coder",
        rs.cfg.coder,
        "--verifier",
        rs.cfg.verifier,
        "--max-iters",
        str(rs.cfg.max_iters),
        "--timeout",
        str(rs.cfg.timeout),
    ]

    # Optional flags (only include if set and you know your agent supports them)
    if rs.cfg.stall is not None:
        cmd += ["--stall", str(rs.cfg.stall)]
    if rs.cfg.no_llm_verifier:
        cmd += ["--no-llm-verifier"]
    if rs.cfg.verify_plugins:
        for p in rs.cfg.verify_plugins:
            cmd += ["--verify-plugin", p]

    push(rs, "info", {"msg": "Starting agent subprocess", "cmd": cmd})

    rs._proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    # Watch for run_dir line, if present
    run_dir_re = re.compile(r"run_dir\s*=\s*(.+)$")

    # Start a side thread to tail events.jsonl (once run_dir exists)
    def tail_events_loop():
        last_dir = None
        while not rs.stop_evt.is_set():
            # Determine run_dir
            rd = None
            if rs.run_dir:
                rd = Path(rs.run_dir)
            else:
                latest = find_latest_run_dir(root)
                if latest:
                    rd = latest
            if rd and rd.exists():
                if last_dir != rd:
                    last_dir = rd
                    rs.run_dir = str(rd)
                    rs._events_pos = 0
                    push(rs, "run_dir", {"run_dir": rs.run_dir})

                events_path = rd / "events.jsonl"
                rs._events_pos, lines = tail_file_lines(events_path, rs._events_pos)
                for line in lines:
                    # attempt parse jsonl, otherwise send raw line
                    try:
                        obj = json.loads(line)
                        push(rs, "event", obj)
                    except Exception:
                        push(rs, "event_raw", line)
            time.sleep(0.25)

    tail_thr = threading.Thread(target=tail_events_loop, daemon=True)
    tail_thr.start()

    # Stream stdout
    try:
        assert rs._proc.stdout is not None
        for line in rs._proc.stdout:
            if line is None:
                break
            line = line.rstrip("\n")
            push(rs, "stdout", line)

            # Try to capture run_dir from stdout
            m = run_dir_re.search(line)
            if m:
                candidate = m.group(1).strip()
                # If relative, interpret from root
                cand_path = Path(candidate)
                if not cand_path.is_absolute():
                    cand_path = (root / candidate).resolve()
                rs.run_dir = str(cand_path)
                push(rs, "run_dir", {"run_dir": rs.run_dir})
    except Exception as e:
        push(rs, "error", {"where": "stdout_stream", "error": str(e)})

    # Wait for process end
    rc = None
    try:
        rc = rs._proc.wait(timeout=60 * 60 * 24)
    except Exception:
        try:
            rs._proc.kill()
        except Exception:
            pass
        rc = -1

    rs.exit_code = rc
    rs.finished_ms = now_ms()
    rs.stop_evt.set()
    push(rs, "done", {"exit_code": rc, "finished_ms": rs.finished_ms})


def start_run(cfg: RunConfig) -> RunState:
    run_id = f"run_{now_ms()}"
    rs = RunState(run_id=run_id, cfg=cfg, started_ms=now_ms())
    with RUNS_LOCK:
        RUNS[run_id] = rs
    t = threading.Thread(target=run_agent_subprocess, args=(rs,), daemon=True)
    t.start()
    return rs


class Handler(BaseHTTPRequestHandler):
    server_version = "ollama-agent-ui/1.0"

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, obj: Any) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(code, "application/json; charset=utf-8", data)

    def _read_json_body(self) -> Any:
        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n) if n > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8", errors="replace"))
        except Exception:
            return {}

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index.html"):
            p = UI_DIR / "index.html"
            self._send(200, "text/html; charset=utf-8", p.read_bytes())
            return

        if self.path.startswith("/api/models"):
            base = self.server.base_ollama  # type: ignore[attr-defined]
            ok, data = list_ollama_models(base)
            self._send_json(200 if ok else 500, data)
            return

        if self.path.startswith("/api/run_status"):
            # /api/run_status?run_id=...
            run_id = None
            if "?" in self.path:
                qs = self.path.split("?", 1)[1]
                for part in qs.split("&"):
                    if part.startswith("run_id="):
                        run_id = part.split("=", 1)[1]
            if not run_id:
                self._send_json(400, {"error": "missing run_id"})
                return
            with RUNS_LOCK:
                rs = RUNS.get(run_id)
            if not rs:
                self._send_json(404, {"error": "unknown run_id"})
                return
            self._send_json(200, {
                "run_id": rs.run_id,
                "started_ms": rs.started_ms,
                "finished_ms": rs.finished_ms,
                "exit_code": rs.exit_code,
                "run_dir": rs.run_dir,
                "cfg": asdict(rs.cfg),
            })
            return

        if self.path.startswith("/api/stream"):
            # SSE: /api/stream?run_id=...
            run_id = None
            if "?" in self.path:
                qs = self.path.split("?", 1)[1]
                for part in qs.split("&"):
                    if part.startswith("run_id="):
                        run_id = part.split("=", 1)[1]
            if not run_id:
                self._send_json(400, {"error": "missing run_id"})
                return

            with RUNS_LOCK:
                rs = RUNS.get(run_id)
            if not rs:
                self._send_json(404, {"error": "unknown run_id"})
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            # Send initial hello
            self.wfile.write(b"event: hello\n")
            self.wfile.write(f"data: {json.dumps({'run_id': run_id})}\n\n".encode("utf-8"))
            self.wfile.flush()

            while True:
                try:
                    msg = rs.q.get(timeout=1.0)
                except queue.Empty:
                    # keepalive
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    if rs.finished_ms is not None:
                        # allow a brief drain time
                        time.sleep(0.5)
                        break
                    continue

                evt = json.dumps(msg, ensure_ascii=False)
                self.wfile.write(b"event: msg\n")
                self.wfile.write(f"data: {evt}\n\n".encode("utf-8"))
                self.wfile.flush()

                if msg.get("t") == "done":
                    break
            return

        self._send(404, "text/plain; charset=utf-8", b"Not found")

    def do_POST(self):
        if self.path == "/api/run":
            body = self._read_json_body()
            cfg = RunConfig(
                ollama=str(body.get("ollama") or self.server.base_ollama),  # type: ignore[attr-defined]
                root=str(body.get("root") or ""),
                task=str(body.get("task") or ""),
                planner=str(body.get("planner") or ""),
                coder=str(body.get("coder") or ""),
                verifier=str(body.get("verifier") or ""),
                max_iters=int(body.get("max_iters") or 12),
                timeout=int(body.get("timeout") or 900),
                stall=body.get("stall", None),
                no_llm_verifier=bool(body.get("no_llm_verifier")) if body.get("no_llm_verifier") is not None else None,
                verify_plugins=body.get("verify_plugins", None),
            )
            if not cfg.root.strip() or not cfg.task.strip() or not cfg.planner or not cfg.coder or not cfg.verifier:
                self._send_json(400, {"error": "root, task, planner, coder, verifier are required"})
                return

            rs = start_run(cfg)
            self._send_json(200, {"run_id": rs.run_id})
            return

        if self.path == "/api/continue":
            body = self._read_json_body()
            root_s = str(body.get("root") or "").strip()
            if not root_s:
                self._send_json(400, {"error": "root is required"})
                return
            root = Path(root_s).expanduser().resolve()
            cfg = load_root_state(root)
            if not cfg:
                self._send_json(404, {"error": f"No {STATE_FILE_NAME} found in root"})
                return
            # Allow overriding ollama URL from UI if provided
            if body.get("ollama"):
                cfg.ollama = str(body.get("ollama"))
            rs = start_run(cfg)
            self._send_json(200, {"run_id": rs.run_id, "loaded_cfg": asdict(cfg)})
            return

        self._send_json(404, {"error": "Not found"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--ollama", default="http://127.0.0.1:11434", help="Default Ollama base URL for UI")
    args = ap.parse_args()

    UI_DIR.mkdir(parents=True, exist_ok=True)
    index_path = UI_DIR / "index.html"
    if not index_path.exists():
        raise SystemExit(f"Missing UI file: {index_path}. Create ui/index.html per instructions.")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.base_ollama = args.ollama  # type: ignore[attr-defined]

    print(f"[ui] Serving on http://{args.host}:{args.port}")
    print(f"[ui] Default Ollama: {args.ollama}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
