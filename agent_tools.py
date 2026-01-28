import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

TEXT_EXT_ALLOW = {".html", ".css", ".js", ".ts", ".json", ".md", ".txt", ".py"}

def sha12_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]

def sha12_text(s: str) -> str:
    return sha12_bytes(s.encode("utf-8", errors="replace"))

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_relpath(root: Path, rel: str) -> Path:
    # prevent path traversal
    target = (root / rel).resolve()
    root_r = root.resolve()
    if not str(target).startswith(str(root_r) + os.sep) and target != root_r:
        raise RuntimeError(f"Refusing to write outside root: {rel}")
    return target

def write_file(root: Path, rel: str, content: str) -> Dict:
    root = root.resolve()
    path = safe_relpath(root, rel)
    ensure_dir(path.parent)

    existed = path.exists()
    old_hash = None
    if existed:
        try:
            old_hash = sha12_bytes(path.read_bytes())
        except Exception:
            old_hash = None

    path.write_text(content, encoding="utf-8")
    new_hash = sha12_bytes(path.read_bytes())
    return {"op": "write_file", "path": str(Path(rel)), "existed": existed, "old_hash": old_hash, "new_hash": new_hash}

def list_tree(root: Path, max_files: int = 400) -> List[str]:
    root = root.resolve()
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(root).as_posix()
        out.append(rel)
        if len(out) >= max_files:
            out.append("... (truncated)")
            break
    return out

def read_text_snippet(root: Path, rel: str, max_chars: int = 6000) -> str:
    p = safe_relpath(root.resolve(), rel)
    if p.suffix.lower() not in TEXT_EXT_ALLOW:
        return ""
    try:
        s = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if len(s) > max_chars:
        return s[:max_chars] + "\n\n... (truncated)"
    return s

def file_exists(root: Path, rel: str) -> bool:
    try:
        p = safe_relpath(root.resolve(), rel)
        return p.exists() and p.is_file()
    except Exception:
        return False

def local_verify_web_grid_project(root: Path) -> Tuple[bool, List[str]]:
    """
    Deterministic verifier for your benchmark:
      - index.html, style.css, script.js exist
      - html references css + js
      - JS uses Math.imul and logs Seed/Checksum
      - JS contains a self-test and legend rendering (either via DOM or innerHTML)
      - CSS contains terrain classes used by JS
    """
    issues = []
    must = ["index.html", "style.css", "script.js"]
    for f in must:
        if not file_exists(root, f):
            issues.append(f"Missing required file: {f}")

    if issues:
        return False, issues

    html = read_text_snippet(root, "index.html", 12000)
    css  = read_text_snippet(root, "style.css", 12000)
    js   = read_text_snippet(root, "script.js", 24000)

    if "style.css" not in html:
        issues.append("index.html does not link style.css")
    if "script.js" not in html:
        issues.append("index.html does not include script.js")

    if "Math.imul" not in js:
        issues.append("script.js missing Math.imul (required for 32-bit FNV-1a)")
    if "FNV" not in js and "fnv" not in js.lower():
        issues.append("script.js missing obvious FNV-1a hash implementation")
    if "Seed:" not in js and "Seed:" not in (js.replace(" ", "")):
        issues.append("script.js missing console log containing 'Seed:'")
    if "Checksum" not in js:
        issues.append("script.js missing console log containing 'Checksum'")
    if "self" not in js.lower() or "PASS" not in js:
        issues.append("script.js missing self-test PASS/FAIL logging")
    if "legend" not in js.lower() and "legend" not in html.lower():
        issues.append("Legend element/function not present")

    # terrain class alignment:
    # We accept either .water/.grass/.mountain OR .cell-water/.cell-grass/.cell-mountain
    js_uses_water = (".water" in js) or ("water" in js.lower())
    css_has_simple = (".water" in css) and (".grass" in css) and (".mountain" in css)
    css_has_cell = (".cell-water" in css) and (".cell-grass" in css) and (".cell-mountain" in css)
    if not (css_has_simple or css_has_cell):
        issues.append("style.css missing terrain classes (water/grass/mountain or cell-water/cell-grass/cell-mountain)")

    return (len(issues) == 0), issues
