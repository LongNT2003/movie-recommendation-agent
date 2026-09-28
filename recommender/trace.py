"""Write compact chat events and a readable transcript shared with MCP."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import json


PROJECT_DIR = Path(__file__).resolve().parent.parent
_current: ContextVar[tuple["EventLog | None", dict]] = ContextVar("movie_trace", default=(None, {}))


def _redact(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if any(word in str(key).lower() for word in
                                     ("api_key", "apikey", "authorization", "password", "secret", "access_token"))
            else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        for key, secret in os.environ.items():
            if len(secret) >= 8 and any(word in key.lower() for word in ("key", "token", "secret", "password")):
                value = value.replace(secret, "[REDACTED]")
    return value


class EventLog:
    def __init__(self, path: Path | None = None):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.path = Path(path) if path is not None else PROJECT_DIR / "logs" / f"chat-{stamp}-{uuid4().hex[:8]}.jsonl"
        self.path = self.path.resolve()
        self.readable_path = self.path.with_suffix(".log")
        if self.readable_path == self.path:
            self.readable_path = self.path.with_name(self.path.name + ".log")

    def write(self, event: str, **fields):
        compact = _compact_event(event, fields)
        if compact is None:
            return
        record = _redact({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **compact,
        })
        data = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode("utf-8")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # One append write per event; never write logs to MCP's stdout transport.
            with self.path.open("ab", buffering=0) as stream:
                stream.write(data)
            with self.readable_path.open("ab", buffering=0) as stream:
                stream.write(_readable_event(record).encode("utf-8"))
        except OSError as exc:
            print(f"Could not write debug log: {type(exc).__name__}", file=sys.stderr)


def _tool_result(result):
    """Keep tool content, without message wrappers or duplicate MCP artifacts."""
    if hasattr(result, "content"):
        content = result.content
    elif isinstance(result, dict) and result.get("type") == "tool":
        content = result.get("content")
    else:
        return result

    def decode(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (ValueError, TypeError):
                pass
        return value

    if isinstance(content, list):
        blocks = [decode(block["text"]) if isinstance(block, dict) and block.get("type") == "text"
                  and "text" in block else block for block in content]
        return blocks[0] if len(blocks) == 1 else blocks
    return decode(content)


def _compact_event(event: str, fields: dict) -> dict | None:
    names = {"chat.start": "user_input", "tool.start": "tool_use",
             "tool.end": "tool_result", "chat.end": "llm_output"}
    if event not in names and not event.endswith(".error"):
        return None
    record = {"event": names.get(event, "error")}
    for key in ("turn_id", "tool_call_id", "tool"):
        if key in fields:
            record[key] = fields[key]
    if event == "chat.start":
        record["input"] = fields["query"]
    elif event == "tool.start":
        record["arguments"] = fields["arguments"]
    elif event == "tool.end":
        result = fields["result"]
        record["result"] = _tool_result(result)
        status = getattr(result, "status", None)
        if status == "error":
            record["status"] = status
    elif event == "chat.end":
        record["output"] = fields["answer"]
    else:
        record["step"] = event
        for key in ("search_id", "error_type", "error"):
            if key in fields:
                record[key] = fields[key]
    return record


def _readable_event(record: dict) -> str:
    label = record["event"].upper().replace("_", " ")
    context = []
    if record.get("turn_id"):
        context.append(f"turn={record['turn_id'][:8]}")
    if record.get("tool"):
        context.append(record["tool"])
    if record.get("tool_call_id"):
        context.append(f"call={record['tool_call_id']}")
    header = f"[{record['timestamp']}] {label}"
    if context:
        header += " | " + " | ".join(context)
    key = {"user_input": "input", "tool_use": "arguments",
           "tool_result": "result", "llm_output": "output"}.get(record["event"])
    if key:
        value = record[key]
        body = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, default=str)
        if record.get("status") == "error":
            body = "Status: error\n" + body
    else:
        body = f"{record['step']}: {record.get('error_type', 'Error')}: {record.get('error', '')}"
        if record.get("search_id"):
            body += f"\nsearch_id={record['search_id']}"
    return f"{header}\n{body}\n\n"


@contextmanager
def trace_context(logger: EventLog | None = None, **fields):
    previous_logger, previous_fields = _current.get()
    token = _current.set((logger or previous_logger, {**previous_fields, **fields}))
    try:
        yield
    finally:
        _current.reset(token)


def log_event(event: str, **fields):
    logger, context = _current.get()
    if logger is None:
        inherited_path = os.getenv("MOVIE_AGENT_TRACE_FILE")
        if not inherited_path:
            return
        logger = EventLog(Path(inherited_path))
    logger.write(event, **{**context, **fields})


def log_exception(event: str, exc: Exception, **fields):
    log_event(event, error_type=type(exc).__name__, error=str(exc), **fields)
