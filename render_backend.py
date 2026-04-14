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
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "claude-sonnet-4-20250514"


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
        self.send_header("Access-Control-Allow-Origin", os.getenv("ALLOWED_ORIGIN", "*"))
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

        try:
            request = self.read_json_body()
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        document = request.get("document")
        prompt = request.get("prompt")
        if not document or not prompt:
            self.send_json(
                {"error": "Document and prompt are required"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        self.stream_playground(request)

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("JSON request body is required")

        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON request body") from exc

        if not isinstance(data, dict):
            raise ValueError("JSON request body must be an object")
        return data

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

        for line in process.stdout:
            if line.strip():
                self.wfile.write(sse_line(line.strip()))
                self.wfile.flush()

        stderr = process.stderr.read()
        if stderr:
            self.wfile.write(sse_line({"type": "stderr", "message": stderr}))

        code = process.wait()
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
