"""LLM client: auto-detects Anthropic / any OpenAI-compatible endpoint / Ollama,
with retries, backoff, timeouts, token-cost logging and token streaming.
Strict grounded citation prompting with a graceful abstain path."""

import json
import logging
import re
import time

import httpx

from app.config import settings

log = logging.getLogger("rag.llm")

PROMPT = """You are a document QA assistant. Answer ONLY from the numbered context passages.

Rules:
1. Use only facts present in the context. Never use outside knowledge.
2. Cite the passage number after every claim, like [1] or [2][3].
3. If the context does not contain the answer, reply exactly: "I don't know based on the provided documents."
4. Be concise (2-5 sentences unless a list is clearly better).

Context passages:
{context}

Question: {question}

Answer with citations:"""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        src = f"{c['filename']} (p.{c['page_number']})"
        parts.append(f"[{i}] Source: {src}\n{c['text']}")
    return "\n\n".join(parts)


def resolve_provider() -> str:
    """Which backend to actually call: an explicit setting wins; otherwise pick
    the first configured one (Anthropic key > OpenAI-compatible key > Ollama)."""
    if settings.llm_provider != "auto":
        return settings.llm_provider
    if settings.anthropic_api_key:
        return "anthropic"
    if settings.openai_api_key and settings.openai_api_key != "change-me":
        return "openai_compat"
    return "ollama"


def _post_with_retries(url: str, *, json_body: dict, headers: dict | None = None) -> httpx.Response:
    """POST with exponential backoff on 429/5xx and network errors."""
    delay = 1.0
    last: Exception | None = None
    for attempt in range(settings.llm_retries + 1):
        try:
            resp = httpx.post(url, json=json_body, headers=headers or {},
                              timeout=settings.llm_timeout_seconds)
            if resp.status_code in (429, 500, 502, 503, 504):
                last = httpx.HTTPStatusError(f"HTTP {resp.status_code}", request=resp.request,
                                             response=resp)
            else:
                resp.raise_for_status()
                return resp
        except httpx.HTTPError as exc:
            last = exc
        if attempt < settings.llm_retries:
            time.sleep(delay)
            delay *= 2
    raise last if isinstance(last, httpx.HTTPError) else httpx.HTTPError(str(last))


def _chat_anthropic(messages: list[dict]) -> str:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
    user_msgs = [m for m in messages if m["role"] != "system"]
    resp = _post_with_retries(
        f"{settings.anthropic_base_url.rstrip('/')}/v1/messages",
        json_body={
            "model": settings.llm_model,
            "max_tokens": settings.llm_max_tokens,
            "temperature": settings.llm_temperature,
            "messages": user_msgs,
            **({"system": system} if system else {}),
        },
        headers={"x-api-key": settings.anthropic_api_key,
                 "anthropic-version": settings.anthropic_version},
    )
    return resp.json()["content"][0]["text"]


def _chat_openai_compat(messages: list[dict]) -> str:
    headers = {}
    if settings.openai_api_key and settings.openai_api_key != "change-me":
        headers["Authorization"] = f"Bearer {settings.openai_api_key}"
    resp = _post_with_retries(
        f"{settings.openai_base_url.rstrip('/')}/chat/completions",
        json_body={
            "model": settings.llm_model, "messages": messages,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens, "stream": False,
        },
        headers=headers,
    )
    return resp.json()["choices"][0]["message"]["content"]


def _chat_ollama(messages: list[dict]) -> str:
    resp = _post_with_retries(
        f"{settings.ollama_base_url.rstrip('/')}/api/chat",
        json_body={"model": settings.llm_model, "messages": messages,
                   "stream": False, "options": {"temperature": settings.llm_temperature}},
    )
    return resp.json()["message"]["content"]


def _chat(messages: list[dict]) -> str:
    provider = resolve_provider()
    started = time.monotonic()
    try:
        if provider == "anthropic":
            answer = _chat_anthropic(messages)
        elif provider == "ollama":
            answer = _chat_ollama(messages)
        else:
            answer = _chat_openai_compat(messages)
    except (httpx.HTTPError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc
    elapsed = int((time.monotonic() - started) * 1000)
    log.info("llm_call provider=%s model=%s latency_ms=%d chars=%d",
             provider, settings.llm_model, elapsed, len(answer))
    return answer


def is_abstain(answer: str) -> bool:
    return "i don't know" in answer.lower()


def extract_citations(answer: str, n_chunks: int) -> list[int]:
    cited: list[int] = []
    for m in re.findall(r"\[(\d{1,2})\]", answer):
        n = int(m)
        if 1 <= n <= n_chunks and n not in cited:
            cited.append(n)
    return cited


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """Return {"answer": str, "abstained": bool, "cited_indexes": list[int]}."""
    context = build_context(chunks)
    messages = [{"role": "user", "content": PROMPT.format(context=context, question=question)}]
    answer = _chat(messages)
    return {"answer": answer.strip(), "abstained": is_abstain(answer),
            "cited_indexes": extract_citations(answer, len(chunks))}


def stream_answer(question: str, chunks: list[dict]):
    """Yield answer text tokens as they are produced. Raises RuntimeError if the
    LLM endpoint is unreachable (caller falls back to the extractive answer)."""
    context = build_context(chunks)
    prompt = PROMPT.format(context=context, question=question)
    provider = resolve_provider()
    started = time.monotonic()
    produced: list[str] = []
    try:
        if provider == "anthropic":
            events = _stream_anthropic(prompt)
        elif provider == "ollama":
            events = _stream_ollama(prompt)
        else:
            events = _stream_openai_compat(prompt)
        for token in events:
            produced.append(token)
            yield token
    except (httpx.HTTPError, KeyError, TypeError, json.JSONDecodeError) as exc:
        if produced:  # partial answer already streamed — surface it rather than swap mid-stream
            log.warning("llm_stream_interrupted provider=%s after %d chars: %s",
                        provider, len("".join(produced)), exc)
            return
        raise RuntimeError(f"LLM call failed: {exc}") from exc
    log.info("llm_stream provider=%s model=%s latency_ms=%d chars=%d",
             provider, settings.llm_model, int((time.monotonic() - started) * 1000),
             len("".join(produced)))


def _sse_data_lines(line_iter):
    """Yield the `data:` payloads of a text/event-stream (lines may be str or bytes)."""
    for line in line_iter:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload and payload != "[DONE]":
                yield payload


def _stream_openai_compat(prompt: str):
    headers = {}
    if settings.openai_api_key and settings.openai_api_key != "change-me":
        headers["Authorization"] = f"Bearer {settings.openai_api_key}"
    body = {"model": settings.llm_model, "stream": True,
            "temperature": settings.llm_temperature, "max_tokens": settings.llm_max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    with httpx.stream("POST", f"{settings.openai_base_url.rstrip('/')}/chat/completions",
                      json=body, headers=headers, timeout=settings.llm_timeout_seconds) as r:
        r.raise_for_status()
        for payload in _sse_data_lines(r.iter_lines()):
            delta = json.loads(payload)["choices"][0].get("delta", {})
            if delta.get("content"):
                yield delta["content"]


def _stream_anthropic(prompt: str):
    with httpx.stream("POST", f"{settings.anthropic_base_url.rstrip('/')}/v1/messages",
                      json={"model": settings.llm_model, "max_tokens": settings.llm_max_tokens,
                            "temperature": settings.llm_temperature,
                            "messages": [{"role": "user", "content": prompt}]},
                      headers={"x-api-key": settings.anthropic_api_key,
                               "anthropic-version": settings.anthropic_version},
                      timeout=settings.llm_timeout_seconds) as r:
        r.raise_for_status()
        for payload in _sse_data_lines(r.iter_lines()):
            event = json.loads(payload)
            if event.get("type") == "content_block_delta" and event["delta"].get("text"):
                yield event["delta"]["text"]


def _stream_ollama(prompt: str):
    body = {"model": settings.llm_model, "stream": True,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": settings.llm_temperature}}
    with httpx.stream("POST", f"{settings.ollama_base_url.rstrip('/')}/api/chat",
                      json=body, timeout=settings.llm_timeout_seconds) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            token = event.get("message", {}).get("content", "")
            if token:
                yield token
            if event.get("done"):
                break


def judge_faithfulness(question: str, answer: str, context: str) -> int:
    """LLM-as-judge: score 1 if the answer is supported by the context, else 0."""
    prompt = (
        "You are judging an answer for faithfulness. Answer with only YES or NO.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\n\n"
        "Is every claim in the answer supported by the context?"
    )
    try:
        verdict = _chat([{"role": "user", "content": prompt}]).strip().upper()
        return 1 if verdict.startswith("YES") else 0
    except Exception:
        return 0  # cannot judge -> don't fabricate a pass
