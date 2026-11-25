import fcntl
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


def _render_value(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, nested in value.items():
            if isinstance(nested, (dict, list)):
                lines.append(f"{prefix}- **{key}**")
                lines.extend(_render_value(nested, indent + 2))
            else:
                lines.append(f"{prefix}- **{key}:** {nested}")
        return lines
    if isinstance(value, list):
        lines = []
        for nested in value:
            if isinstance(nested, (dict, list)):
                lines.extend(_render_value(nested, indent))
            else:
                lines.append(f"{prefix}- {nested}")
        return lines
    return [f"{prefix}{value}"]


def _render_text(value: Any) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
        if isinstance(parsed, (dict, list)):
            return "\n".join(_render_value(parsed))
        return value
    return "\n".join(_render_value(value))


class MarkdownLLMTranscript(BaseCallbackHandler):
    def __init__(self, path: str):
        self.path = Path(path).expanduser()

    def _append(self, entry: str) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self.path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            with os.fdopen(descriptor, "a", encoding="utf-8") as transcript:
                fcntl.flock(transcript.fileno(), fcntl.LOCK_EX)
                transcript.write(entry)
                transcript.flush()
                fcntl.flock(transcript.fileno(), fcntl.LOCK_UN)
        except OSError as error:
            logger.warning("Could not append LLM transcript: %s", type(error).__name__)

    @staticmethod
    def _header(kind: str, run_id: Any) -> str:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return f"## {kind} ({timestamp}, run `{run_id}`)\n\n"

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        model = (serialized or {}).get("kwargs", {}).get("model") or "unknown model"
        entry = self._header(f"Request: {model}", run_id)
        for batch_index, batch in enumerate(messages, start=1):
            if len(messages) > 1:
                entry += f"### Message batch {batch_index}\n\n"
            for message in batch:
                role = getattr(message, "type", "message").capitalize()
                entry += f"### {role}\n\n{_render_text(message.content)}\n\n"
        self._append(entry)

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        entry = self._header("Response", run_id)
        for generations in getattr(response, "generations", []):
            for generation in generations:
                message = getattr(generation, "message", None)
                if message is None:
                    entry += f"{_render_text(getattr(generation, 'text', ''))}\n\n"
                    continue
                content = getattr(message, "content", "")
                if content:
                    entry += f"{_render_text(content)}\n\n"
                for tool_call in getattr(message, "tool_calls", []):
                    entry += (
                        f"### Structured response: {tool_call.get('name', 'tool')}\n\n"
                    )
                    entry += "\n".join(_render_value(tool_call.get("args", {})))
                    entry += "\n\n"
        self._append(entry)
