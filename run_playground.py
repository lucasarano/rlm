"""
Playground runner for the RLM visualizer.

Runs an RLM completion with a streaming logger that writes each iteration
as a JSONL line to stdout, enabling real-time SSE streaming to the browser.

Events emitted (all JSON lines on stdout):
  - progress:      Status messages during setup
  - metadata:      RLM configuration metadata
  - response_text: The model's raw response text (before code execution)
  - code_result:   Result of a single code block execution
  - iteration:     Complete iteration data (for persistence / final rendering)
  - done:          Completion summary
  - error:         Fatal error
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from datetime import datetime

from dotenv import load_dotenv

from rlm import RLM
from rlm.core.types import EnvironmentType, RLMIteration, RLMMetadata
from rlm.logger.rlm_logger import RLMLogger

SUBCALL_PATTERN = re.compile(r"\b(?:llm_query|rlm_query|llm_query_batched|rlm_query_batched)\s*\(")
_EMIT_LOCK = threading.Lock()
ISOLATED_ENVIRONMENTS = {"modal", "prime", "daytona", "e2b"}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def env_float(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return float(value)


def clamp(value: int, *, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def load_environment_kwargs(raw: str | None) -> dict:
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("RLM_ENVIRONMENT_KWARGS must be a JSON object")
    return parsed


def resolve_execution_environment(requested_environment: str) -> tuple[EnvironmentType, dict]:
    environment = requested_environment.strip().lower()
    supported = {"local", "docker", *ISOLATED_ENVIRONMENTS}
    if environment not in supported:
        raise ValueError(f"Unsupported RLM execution environment: {requested_environment}")

    is_production = env_flag("RLM_PRODUCTION", default=os.getenv("NODE_ENV") == "production")
    allow_local = env_flag("RLM_ALLOW_LOCAL_EXEC", default=False)
    if is_production and environment not in ISOLATED_ENVIRONMENTS and not allow_local:
        raise RuntimeError(
            "Refusing to run user-triggered code in a non-isolated environment. "
            "Set RLM_EXEC_ENVIRONMENT to modal, prime, daytona, or e2b, or set "
            "RLM_ALLOW_LOCAL_EXEC=1 only for a trusted private deployment."
        )

    return environment, load_environment_kwargs(os.getenv("RLM_ENVIRONMENT_KWARGS"))


def resolve_log_dir() -> str | None:
    explicit_log_dir = os.getenv("RLM_LOG_DIR")
    if explicit_log_dir:
        return explicit_log_dir

    if env_flag("RLM_ENABLE_PUBLIC_LOGS", default=False):
        return os.path.join(os.path.dirname(__file__), "visualizer", "public", "logs")

    return None


def emit(event: dict) -> None:
    """Write a JSON event to stdout and flush immediately."""
    with _EMIT_LOCK:
        sys.stdout.write(json.dumps(event) + "\n")
        sys.stdout.flush()


def emit_progress(message: str, detail: str | None = None) -> None:
    event: dict = {"type": "progress", "message": message, "timestamp": datetime.now().isoformat()}
    if detail:
        event["detail"] = detail
    emit(event)


def emit_debug(message: str, **details) -> None:
    """Emit low-volume structured debug data for live-stream troubleshooting."""
    emit(
        {
            "type": "debug",
            "source": "run_playground",
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }
    )


class StreamingLogger(RLMLogger):
    """Logger that writes each iteration to stdout as JSONL for SSE streaming."""

    def __init__(self, log_dir: str | None = None, file_name: str = "rlm"):
        super().__init__(log_dir=log_dir, file_name=file_name)

    def log_metadata(self, metadata: RLMMetadata) -> None:
        super().log_metadata(metadata)
        emit(
            {
                "type": "metadata",
                "timestamp": datetime.now().isoformat(),
                **metadata.to_dict(),
            }
        )

    def log(self, iteration: RLMIteration) -> None:
        super().log(iteration)
        emit_debug(
            "iteration_logged",
            response_chars=len(iteration.response or ""),
            response_subcall_patterns=len(SUBCALL_PATTERN.findall(iteration.response or "")),
            code_blocks=len(iteration.code_blocks),
            executed_subcalls=sum(
                len(code_block.result.rlm_calls)
                for code_block in iteration.code_blocks
                if code_block.result is not None
            ),
        )
        emit(
            {
                "type": "iteration",
                "iteration": self._iteration_count,
                "timestamp": datetime.now().isoformat(),
                **iteration.to_dict(),
            }
        )


# --- Callbacks for real-time granular streaming ---

_iteration_start_time: float = 0.0


def on_iteration_start(depth: int, iteration_num: int) -> None:
    global _iteration_start_time
    _iteration_start_time = time.perf_counter()
    emit_progress(f"Starting iteration {iteration_num + 1}", f"depth={depth}")


def on_iteration_complete(depth: int, iteration_num: int, duration: float) -> None:
    emit_progress(f"Iteration {iteration_num + 1} complete", f"{duration:.1f}s")


def on_response(depth: int, response_text: str) -> None:
    """Emit the full response text immediately so the frontend can render it
    before code blocks finish executing."""
    emit_debug(
        "response_text_callback",
        depth=depth,
        response_chars=len(response_text or ""),
        response_subcall_patterns=len(SUBCALL_PATTERN.findall(response_text or "")),
    )
    emit(
        {
            "type": "response_text",
            "depth": depth,
            "text": response_text,
            "timestamp": datetime.now().isoformat(),
        }
    )


def on_token(depth: int, text: str) -> None:
    """Emit a token delta so the frontend can render it incrementally."""
    if depth > 0:
        emit_debug("token", depth=depth, token_chars=len(text))
    emit({"type": "token", "text": text, "depth": depth, "timestamp": datetime.now().isoformat()})


def on_subcall_start(depth: int, model: str, prompt_preview: str) -> None:
    """Emit a structured subcall_start event for live sub-thread navigation."""
    emit_debug(
        "subcall_start_callback",
        depth=depth,
        model=model,
        prompt_preview_chars=len(prompt_preview or ""),
    )
    emit(
        {
            "type": "subcall_start",
            "depth": depth,
            "model": model,
            "prompt_preview": prompt_preview[:200] if prompt_preview else "",
            "timestamp": datetime.now().isoformat(),
        }
    )
    emit_progress("Sub-LM call started", f"depth={depth} model={model}")


def on_subcall_complete(depth: int, model: str, duration: float, error: str | None) -> None:
    """Emit a structured subcall_complete event when sub-thread finishes."""
    emit_debug(
        "subcall_complete_callback",
        depth=depth,
        model=model,
        duration=duration,
        error=error,
    )
    emit(
        {
            "type": "subcall_complete",
            "depth": depth,
            "model": model,
            "duration": duration,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }
    )
    status = f"error: {error}" if error else f"{duration:.1f}s"
    emit_progress("Sub-LM call finished", f"depth={depth} model={model} {status}")


def main():
    parser = argparse.ArgumentParser(description="Run RLM playground query")
    parser.add_argument("--document-path", required=True, help="Path to document file")
    parser.add_argument("--prompt", required=True, help="User prompt/question")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Model name")
    parser.add_argument("--max-iterations", type=int, default=15, help="Max iterations")
    parser.add_argument("--max-depth", type=int, default=2, help="Max recursive subcall depth")
    parser.add_argument(
        "--environment",
        default=os.getenv("RLM_EXEC_ENVIRONMENT", "local"),
        help="Execution environment: local, docker, modal, prime, daytona, or e2b",
    )
    args = parser.parse_args()

    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-key-here":
        emit({"type": "error", "message": "ANTHROPIC_API_KEY not set in .env file"})
        sys.exit(1)

    emit_progress("Loading document...")

    with open(args.document_path) as f:
        document_content = f.read()

    emit_progress(f"Document loaded ({len(document_content):,} chars)")

    max_iterations = clamp(
        args.max_iterations,
        minimum=1,
        maximum=env_int("RLM_MAX_ITERATIONS", 8),
    )
    max_depth = clamp(args.max_depth, minimum=1, maximum=env_int("RLM_MAX_DEPTH", 2))
    max_timeout = env_float("RLM_MAX_TIMEOUT_SECONDS")
    max_tokens = os.getenv("RLM_MAX_TOKENS")
    environment, environment_kwargs = resolve_execution_environment(args.environment)
    log_dir = resolve_log_dir()
    logger = StreamingLogger(log_dir=log_dir)

    emit_progress(
        "Creating RLM instance...",
        f"model={args.model} environment={environment} max_iterations={max_iterations}",
    )

    rlm = RLM(
        backend="anthropic",
        backend_kwargs={
            "model_name": args.model,
            "api_key": api_key,
        },
        environment=environment,
        environment_kwargs=environment_kwargs,
        max_depth=max_depth,
        max_iterations=max_iterations,
        max_timeout=max_timeout,
        max_tokens=int(max_tokens) if max_tokens else None,
        max_errors=env_int("RLM_MAX_ERRORS", 3),
        logger=logger,
        verbose=False,
        on_iteration_start=on_iteration_start,
        on_iteration_complete=on_iteration_complete,
        on_response=on_response,
        on_token=on_token,
        on_subcall_start=on_subcall_start,
        on_subcall_complete=on_subcall_complete,
    )

    emit_progress("Starting completion...", "Waiting for first model response")

    result = rlm.completion(document_content, root_prompt=args.prompt)

    emit(
        {
            "type": "done",
            "response": result.response,
            "execution_time": result.execution_time,
            "usage_summary": result.usage_summary.to_dict() if result.usage_summary else None,
        }
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
        sys.exit(1)
