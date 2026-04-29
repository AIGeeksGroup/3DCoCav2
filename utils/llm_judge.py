import json
import os
import re
import urllib.request
from typing import List, Optional


def _post_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_index(text: str, num_candidates: int) -> Optional[int]:
    match = re.search(r"\b(\d+)\b", text)
    if not match:
        return None
    idx = int(match.group(1))
    if 1 <= idx <= num_candidates:
        return idx - 1
    return None


def select_best_caption(
    summary: str,
    candidates: List[str],
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float = 0.0,
    timeout: int = 30,
) -> Optional[int]:
    """
    Select best caption index using OpenAI-compatible Chat Completions API.
    Returns index (0-based) or None on failure.
    """
    prompt = (
        "You are judging 3D scene captions. "
        "Given the scene summary and candidate captions, "
        "choose the single best caption that is most faithful to the summary "
        "and avoids hallucinations. "
        "Return ONLY the index number (1-based)."
    )
    candidates_text = "\n".join(
        [f"{i+1}. {cap}" for i, cap in enumerate(candidates)]
    )
    user_text = f"Scene summary:\n{summary}\n\nCandidates:\n{candidates_text}\n\nAnswer:"

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": temperature,
        "max_tokens": 10,
    }

    try:
        response = _post_json(base_url, payload, headers, timeout)
        content = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return _parse_index(content, len(candidates))
    except Exception:
        return None
