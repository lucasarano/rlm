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
from rlm.core.types import RLMIteration, RLMMetadata
from rlm.logger.rlm_logger import RLMLogger

SUBCALL_PATTERN = re.compile(r"\b(?:llm_query|rlm_query|llm_query_batched|rlm_query_batched)\s*\(")
_EMIT_LOCK = threading.Lock()


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

    log_dir = os.path.join(os.path.dirname(__file__), "visualizer", "public", "logs")
    logger = StreamingLogger(log_dir=log_dir)

    emit_progress("Creating RLM instance...", f"model={args.model}")

    rlm = RLM(
        backend="anthropic",
        backend_kwargs={
            "model_name": args.model,
            "api_key": api_key,
        },
        environment="local",
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
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
    main()
