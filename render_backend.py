"""Render backend for the RLM visualizer playground.

This service exposes the same SSE contract as visualizer/src/app/api/run/route.ts
and delegates execution to run_playground.py in a normal Python environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_BODY_BYTES = int(os.getenv("RLM_MAX_BODY_BYTES", str(2 * 1024 * 1024)))
MAX_DOCUMENT_CHARS = int(os.getenv("RLM_MAX_DOCUMENT_CHARS", "500000"))
MAX_PROMPT_CHARS = int(os.getenv("RLM_MAX_PROMPT_CHARS", "8000"))
MAX_ITERATIONS = int(os.getenv("RLM_MAX_ITERATIONS", "8"))
MAX_DEPTH = int(os.getenv("RLM_MAX_DEPTH", "2"))
RUN_TIMEOUT_SECONDS = int(os.getenv("RLM_RUN_TIMEOUT_SECONDS", "300"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RLM_RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RLM_RATE_LIMIT_MAX_REQUESTS", "20"))
RATE_LIMIT_BUCKETS: dict[str, tuple[int, float]] = {}
RATE_LIMIT_LOCK = threading.Lock()


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def clamp_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def sse_line(event: dict[str, Any] | str) -> bytes:
    if isinstance(event, str):
        payload = event
    else:
        payload = json.dumps(event)
    return f"data: {payload}\n\n".encode()


class RLMBackendHandler(BaseHTTPRequestHandler):
    server_version = "RLMRenderBackend/1.0"

    def log_message(self, format: str, *args: object) -> None:
        message = format % args
        sys.stderr.write(f"{self.address_string()} - - [{self.log_date_time_string()}] {message}\n")

    def end_headers(self) -> None:
        allowed_origin = os.getenv("ALLOWED_ORIGIN")
        if allowed_origin:
            self.send_header("Access-Control-Allow-Origin", allowed_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Max-Age", "86400")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json({"status": "ok"})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        if not self.is_authorized():
            self.send_json({"error": "Unauthorized"}, status=HTTPStatus.UNAUTHORIZED)
            return

        retry_after = self.rate_limit_retry_after()
        if retry_after is not None:
            self.send_json(
                {"error": "Rate limit exceeded", "retry_after": retry_after},
                status=HTTPStatus.TOO_MANY_REQUESTS,
            )
            return

        try:
            request = self.read_json_body()
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            request = self.validate_run_request(request)
        except ValueError as exc:
            self.send_json(
                {"error": str(exc)},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        self.stream_playground(request)

    def is_authorized(self) -> bool:
        token = os.getenv("RLM_RUN_API_TOKEN")
        allow_unauthenticated = env_flag("RLM_ALLOW_UNAUTHENTICATED_RUNS", default=False)
        if not token:
            return allow_unauthenticated

        auth_header = self.headers.get("Authorization", "")
        bearer = f"Bearer {token}"
        return auth_header == bearer or self.headers.get("X-RLM-Run-Token") == token

    def rate_limit_retry_after(self) -> int | None:
        forwarded_for = self.headers.get("X-Forwarded-For", "")
        client_ip = forwarded_for.split(",")[0].strip() or self.client_address[0]
        now = time.time()

        with RATE_LIMIT_LOCK:
            count, reset_at = RATE_LIMIT_BUCKETS.get(
                client_ip, (0, now + RATE_LIMIT_WINDOW_SECONDS)
            )
            if reset_at <= now:
                RATE_LIMIT_BUCKETS[client_ip] = (1, now + RATE_LIMIT_WINDOW_SECONDS)
                return None
            if count >= RATE_LIMIT_MAX_REQUESTS:
                return max(1, int(reset_at - now))

            RATE_LIMIT_BUCKETS[client_ip] = (count + 1, reset_at)
            return None

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("JSON request body is required")
        if length > MAX_BODY_BYTES:
            raise ValueError(f"Request body exceeds {MAX_BODY_BYTES} byte limit")

        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON request body") from exc

        if not isinstance(data, dict):
            raise ValueError("JSON request body must be an object")
        return data

    def validate_run_request(self, request: dict[str, Any]) -> dict[str, Any]:
        document = request.get("document")
        prompt = request.get("prompt")
        if not isinstance(document, str) or not document.strip():
            raise ValueError("Document is required")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt is required")
        if len(document) > MAX_DOCUMENT_CHARS:
            raise ValueError(f"Document exceeds {MAX_DOCUMENT_CHARS} character limit")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"Prompt exceeds {MAX_PROMPT_CHARS} character limit")

        model = request.get("model") or DEFAULT_MODEL
        if not isinstance(model, str):
            raise ValueError("Model must be a string")

        return {
            "document": document,
            "prompt": prompt,
            "model": model,
            "maxIterations": clamp_int(
                request.get("maxIterations"),
                default=min(5, MAX_ITERATIONS),
                minimum=1,
                maximum=MAX_ITERATIONS,
            ),
            "maxDepth": clamp_int(
                request.get("maxDepth"),
                default=min(2, MAX_DEPTH),
                minimum=1,
                maximum=MAX_DEPTH,
            ),
        }

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def stream_playground(self, request: dict[str, Any]) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        doc_path = self.write_document(str(request["document"]))
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "run_playground.py"),
                "--document-path",
                str(doc_path),
                "--prompt",
                str(request["prompt"]),
                "--model",
                str(request.get("model") or DEFAULT_MODEL),
                "--max-iterations",
                str(request.get("maxIterations") or 15),
                "--max-depth",
                str(request.get("maxDepth") or 2),
                "--environment",
                os.getenv("RLM_EXEC_ENVIRONMENT", "local"),
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        try:
            self.forward_stream(process)
        finally:
            if process.poll() is None:
                process.kill()
            try:
                doc_path.unlink()
            except FileNotFoundError:
                pass

    def write_document(self, document: str) -> Path:
        tmp_dir = Path(tempfile.gettempdir()) / "rlm-playground"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        doc_path = tmp_dir / f"doc_{int(time.time() * 1000)}.txt"
        doc_path.write_text(document, encoding="utf-8")
        return doc_path

    def forward_stream(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        assert process.stderr is not None

        timed_out = threading.Event()

        def kill_on_timeout() -> None:
            if process.poll() is None:
                timed_out.set()
                process.kill()

        timer = threading.Timer(RUN_TIMEOUT_SECONDS, kill_on_timeout)
        timer.daemon = True
        timer.start()

        for line in process.stdout:
            if line.strip():
                self.wfile.write(sse_line(line.strip()))
                self.wfile.flush()

        timer.cancel()
        stderr = process.stderr.read()
        if stderr:
            self.wfile.write(sse_line({"type": "stderr", "message": stderr}))

        code = process.wait()
        if timed_out.is_set():
            self.wfile.write(
                sse_line(
                    {
                        "type": "error",
                        "message": f"Run exceeded {RUN_TIMEOUT_SECONDS}s timeout",
                    }
                )
            )
            code = -1

        if code != 0:
            self.wfile.write(
                sse_line({"type": "error", "message": f"Process exited with code {code}"})
            )

        self.wfile.write(sse_line("[DONE]"))
        self.wfile.flush()


def main() -> None:
    port = int(os.getenv("PORT", "10000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), RLMBackendHandler)
    print(f"RLM backend listening on port {port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
