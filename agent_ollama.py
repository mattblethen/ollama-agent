import json
import time
import urllib.request
import urllib.error

def _post_json(url: str, payload: dict, timeout_s: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except Exception:
        # Ollama sometimes returns plain text on failure proxies; wrap it
        return {"raw": raw}

def ollama_tags(base_url: str, timeout_s: int = 15) -> dict:
    url = base_url.rstrip("/") + "/api/tags"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)

def ollama_chat(base_url: str, model: str, messages: list, temperature: float,
                timeout_s: int, retries: int = 2, stall_cb=None) -> str:
    """
    Non-streaming chat. Retries with backoff. stall_cb() called periodically while waiting.
    """
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            # We can't tick while urlopen blocks, so we rely on timeout_s + retries.
            out = _post_json(url, payload, timeout_s=timeout_s)
            if "message" in out and isinstance(out["message"], dict):
                return out["message"].get("content", "") or ""
            if "raw" in out:
                return out["raw"]
            return json.dumps(out, ensure_ascii=False)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
        except Exception as e:
            last_err = e

        # backoff
        sleep_s = min(2 ** attempt, 8)
        if stall_cb:
            try:
                stall_cb(note=f"ollama_chat retry backoff {sleep_s}s (attempt {attempt+1}/{retries+1})")
            except Exception:
                pass
        time.sleep(sleep_s)

    raise RuntimeError(f"Ollama chat failed after {retries+1} attempts: {last_err}")
